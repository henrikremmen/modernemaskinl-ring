"""Serialize notebook 18 only; no data processing or model training."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def cell(kind,source,ident):
    result=dict(cell_type=kind,id=ident,metadata={},source=source.strip().splitlines(keepends=True))
    if kind=='code':
        ast.parse(source.strip())
        result.update(execution_count=None,outputs=[])
    return result


def embed(name):
    return '\n'.join(line for line in (ROOT/name).read_text().splitlines() if not line.startswith('from src.'))


old=json.loads((ROOT/'notebooks/17_physics_feature_refinement.ipynb').read_text())
old_config=''.join(next(c['source'] for c in old['cells'] if c.get('id')=='config'))
v4_digest=re.search(r"SOURCE_DIGEST = '([^']+)'",old_config).group(1)
sources={
    'pipeline':embed('src/sintef_pipeline.py'),
    'physics':embed('src/sintef_research_features.py'),
    'fit_helpers':embed('src/sintef_challenger_runner.py').split('class ChallengerRunner:')[0],
    'competition':embed('src/sintef_feature_refinement.py').split('def screen_refinements(')[0],
    'paths':embed('src/sintef_path_features.py'),
    'runner':embed('src/sintef_path_runner.py'),
}
digest=hashlib.sha256(''.join(sources.values()).encode()).hexdigest()
resources='''# FIRST CODE CELL: execution resources only; identical statistical settings in both profiles.
import os
RUN_PROFILE = 'local'  # 'professor': <=4 CPU threads, no restored predictions
LOCAL_CPU_THREADS = 12  # Try 8 if the Mac is unresponsive; 12 is not benchmarked as optimal.
if RUN_PROFILE not in ('local','professor'):
    raise ValueError('Unknown profile')
THREADS = min(LOCAL_CPU_THREADS if RUN_PROFILE=='local' else 4, os.cpu_count() or 1)
if THREADS < 1:
    raise ValueError('Need at least one thread')
USE_CHECKPOINTS = RUN_PROFILE=='local'
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['OMP_NUM_THREADS']=str(THREADS)
print(f'{RUN_PROFILE}: {THREADS} CPU threads; one model at a time; restore={USE_CHECKPOINTS}')
'''
config=old_config.split('SPECS =')[0]
config=re.sub(r"SOURCE_DIGEST = '[^']+'",f"SOURCE_DIGEST = '{digest}'\nV4_SOURCE_DIGEST = '{v4_digest}'",config)
config+='''ROUNDS = {'xgb':BASELINE_ROUNDS['xgb_v2'], 'lgb':BASELINE_ROUNDS['lgb_v2']}
SPECS = {
    'xgb':dict(backend='xgboost',features='competition',max_rounds=2000,parameters=xgb_parameters),
    'lgb':dict(backend='lightgbm',features='competition',max_rounds=1800,parameters=lgb_parameters),
}
'''
build='''# Input-only feature construction. No estimators are fitted in this cell.
data=build_project_data(ROOT)
extra,local,mapping=build_research_features(data)
additions,competition_audit=build_refinement_features(data)
competition=additions['competition']
del additions  # Do not retain the rejected competition+timing variant.
paths,path_audit=build_path_features(data,local)
runner=PathRunner(data,extra,local,competition,paths,SPECS,ROUNDS,THREADS,
                  SOURCE_DIGEST,V4_SOURCE_DIGEST,USE_CHECKPOINTS)
path_audit.to_csv(runner.directory/'feature_manifest.csv',index=False)
display(path_audit)
print('Training cases:',data.n_train_cases,'test cases:',data.n_test_cases)
print('New path-feature arrays (MB):',round(sum(f.memory_usage().sum() for f in paths.values())/1e6,1))
print('Research: 56 development unit fits, <=56 audit fits, 28 final fits.')
print('Each development fit provides three tree-count snapshots; not three separate trainings.')
'''
development='''# TRAINING A — select features, round multiplier, and one global blend weight on 2021.
selection,scores=develop_paths(runner)
display(scores.sort_values('micro_auc',ascending=False))
print('Chosen configuration:',selection['members'],'XGB weight:',selection['xgb_weight'])
print('Development blend:',selection['development_auc'],'v4 control:',selection['development_baseline_auc'])
'''
audit='''# TRAINING B — frozen 2022 historical audit; no retuning against these labels.
# Restart safely: run resource/config/definition/build cells, then this cell.
selection=json.loads((runner.directory/'selection.json').read_text())
selection=audit_paths(runner,selection)
display(pd.read_csv(runner.directory/'audit_diagnostics.csv').query("group == 'unit'"))
'''
final='''# TRAINING C — selected pair on all labeled cases, then actual submission CSVs.
selection=json.loads((runner.directory/'selection.json').read_text())
if 'audit_auc' not in selection:
    raise ValueError('Run TRAINING B before final training')
if not selection['passed_audit']:
    print('WARNING: candidate did not beat v4 on 2022; this export is experimental, not a proven upgrade.')
submission_path,submission=final_paths(runner,selection)
display(submission.iloc[:3,:8])
print('Total minutes:',round((time.perf_counter()-NOTEBOOK_STARTED)/60,1))
'''
export=r'''# Self-contained, final-only notebook. No old artifacts/src files required when grading.
document=json.loads((ROOT/'notebooks/18_multiscale_boosting.ipynb').read_text())
keep={'resources','config','pipeline','physics','fit_helpers','competition','paths','runner','build'}
short_cells=[]
for c in document['cells']:
    if c.get('id') not in keep:
        continue
    c=json.loads(json.dumps(c)); c['execution_count']=None; c['outputs']=[]
    source=''.join(c['source'])
    if c['id']=='resources':
        source=source.replace("RUN_PROFILE = 'local'", "RUN_PROFILE = 'professor'")
    if c['id']=='runner':
        source=source.split('def select_pair(')[0]+'\n\ndef final_paths('+source.split('def final_paths(',1)[1]
    if c['id']=='build':
        source='\n'.join(line for line in source.splitlines() if not line.startswith("print('Research:") and not line.startswith("print('Each development"))
    c['source']=source.splitlines(keepends=True);short_cells.append(c)
intro='# Short notebook — multiscale boosted blend\n\n**Henrik Remmen (541747), Group 119 — [119] SINTEF Databreach**\n\nRun all cells offline beside the supplied raw data and mapping/sample CSVs. Four CPU threads; no restored model predictions. Hyperparameters selected in notebook 18 on 2021, checked on reused 2022 history. Only 28 final unit fits; no development search. Actual cold runtime must be measured against 12 hours / 32 GB.\n'
short_cells.insert(0,dict(cell_type='markdown',id='intro',metadata={},source=intro.splitlines(keepends=True)))
source='selection = '+repr(selection)+'\npath, submission = final_paths(runner, selection)\nprint("Elapsed minutes:",(time.perf_counter()-NOTEBOOK_STARTED)/60)\n'
short_cells.append(dict(cell_type='code',id='final',metadata={},execution_count=None,outputs=[],source=source.splitlines(keepends=True)))
document['cells']=short_cells
short_path=runner.directory/'Short_notebook_multiscale_v5.ipynb'
short_path.write_text(json.dumps(document,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print('Professor notebook:',short_path)
print('Upload candidate:',submission_path)
'''
intro='''# 18 — Whole-week features and staged XGBoost/LightGBM blend

**Henrik Remmen (541747), Group 119 — [119] SINTEF Databreach**

Current reported best: notebook 17 blend **0.99005** (XGB 0.98996, LGBM 0.98968).
This is a new experiment, not a guarantee of beating 0.99562. Keep your current best CSV.

## Hypotheses and controls

1. The successful models reduce the price/inflow trajectory to many local summaries.
   Add each day's price mean/peak relative to the current hour, six fixed DCT shape
   coefficients, contiguous competing-price runs, each day's profitable/unprofitable
   margin and source inflow, and the concentration/timing of profitable opportunities.
   These are 54 input-only features per plant. No future observed storage or labels.
2. The old per-unit tree counts were frozen from earlier experiments. They may underfit
   the richer features/larger training history. Fit up to 2x those counts, retaining
   predictions at 1x, 1.5x and 2x. Keep depth/leaves, learning rate, regularization and
   seeds unchanged. This distinguishes extra boosting from the new feature group.

2021 selects the feature set and multiplier for each backend, then one global XGB
weight from {0.25, 0.5, 0.75}. 2022 checks that frozen pair against the exact v4 50/50
control. No per-generator weight search, no label smoothing, no hard ON/OFF output.
Both years have informed earlier development, so neither is a pristine holdout.
Daily case horizons are purged at year boundaries. Public scores do not enter selection.

## Run

Restart the kernel and run all cells. A = development, B = historical audit,
C = final training and CSV creation. Final cell exports a self-contained professor
notebook. That short notebook embeds all code and trains from raw data, without
requiring any models, artifacts, `src/`, or other notebooks. It always uses <=4 CPU
threads, with cache reads disabled. Use the project's pinned requirements.

Local mode defaults to 12 threads, sequential float32 fits, BLAS one thread.
Completed per-generator tasks resume after an interruption; the interrupted fit
restarts. 2022/final v4 controls can reuse notebook-17 predictions only if raw data,
source signature, parameters, rounds, case IDs and package versions match.
Native notebook-18 checkpoints also include thread count and exact source signature.
Changing thread count invalidates those native caches to avoid hidden assumptions.

Research has <=140 unit fits, not 420: each development fit yields three prefixes.
Final-only reproduction has exactly 28 fits, at most twice the old tree counts.
No GPU or parallel model copies. Training runtime/peak RAM have NOT been benchmarked;
cold-test the selected short notebook on four cores against 12 hours / 32 GB.
If the audit fails, the new CSV remains explicitly experimental; it does not silently
replace itself with the older blend. A selected 1x v4 pair can legitimately reproduce
the old model if the new alternatives lose.

Only small diagnostic probes and tiny API tests were run while building this notebook.
No full-scale candidate training was performed by the assistant.
'''
cells=[cell('markdown',intro,'intro'),cell('code',resources,'resources'),cell('code',config,'config')]
for name,source in sources.items():
    cells.extend([cell('markdown','## '+name+' — definitions only',name+'-heading'),cell('code',source,name)])
for ident,title,source in [('build','Build features — no training',build),('development','TRAINING A — controlled development',development),('audit','TRAINING B — frozen historical audit',audit),('final','TRAINING C — submission files',final),('export','Export professor notebook',export)]:
    cells.extend([cell('markdown','## '+title,ident+'-heading'),cell('code',source,ident)])
cells.append(cell('markdown','''## Files and interpretation

- Upload candidate: `submissions/multiscale_boosting_v5.csv`.
- Individual models: `submissions/xgb_multiscale_v5.csv`, `submissions/lgb_multiscale_v5.csv`.
- Selection, feature manifest, development scores, 2022 diagnostics, per-unit gains and
  runtimes: `artifacts/multiscale_boosting_v5/`.
- Professor version: `artifacts/multiscale_boosting_v5/Short_notebook_multiscale_v5.ipynb`.

Inspect `development_scores.csv`: compare 1x v4 against 1x path to isolate features,
then compare 1x/1.5x/2x within each variant to assess additional boosting. Gain rankings
are descriptive, not causal proof of a feature's value. Check audit errors by unit,
month and horizon day. Do not claim a leaderboard improvement before submitting.

Research details: `RESEARCH_MULTISCALE_V5.md` in the development workspace.

Primary references: [XGBoost tuning](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html),
[LightGBM tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html),
[SINTEF generator physics](https://docs.shop.sintef.energy/objects/generator/generator.html),
[scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html).
These motivate hypotheses and safeguards; they do not establish performance on this task.
''','outputs'))
nb=dict(cells=cells,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3'),language_info=dict(name='python')),nbformat=4,nbformat_minor=5)
path=ROOT/'notebooks/18_multiscale_boosting.ipynb'
path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print(path)


if __name__=='__main__':
    pass
