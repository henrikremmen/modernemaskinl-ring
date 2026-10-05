"""Build notebook 17 without executing any data processing or model fitting."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, text, ident):
    c = dict(cell_type=kind, id=ident, metadata={}, source=text.strip().splitlines(keepends=True))
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def embed(path):
    return '\n'.join(l for l in (ROOT/path).read_text().splitlines() if not l.startswith('from src.'))


pipeline = embed('src/sintef_pipeline.py')
physics = embed('src/sintef_research_features.py')
runner = embed('src/sintef_challenger_runner.py').split('def screen_challenger(')[0]
runner = runner.replace("'artifacts/challenger_v3'", "'artifacts/physics_features_v4'")
runner = runner.replace("if spec['features'] == 'dispatch':", "if spec['features'] != 'physics':")
runner = runner.replace("self.additions[plant]", "self.additions[spec['features']][plant]")
features = embed('src/sintef_feature_refinement.py')
digest = hashlib.sha256((pipeline+physics+runner+features).encode()).hexdigest()
# Scheduling controls do not alter statistical settings or the feature code.
# Keep the pre-resource-change digest and canonical four-thread cache identity
# so already completed notebook-17 tasks remain reusable in local mode.
runner = runner.replace('specs, min(4, int(threads))', 'specs, max(1, min(int(threads), os.cpu_count() or 1))')
runner = runner.replace('json.dumps(manifest, sort_keys=True)', 'json.dumps(dict(manifest, threads=4), sort_keys=True)')
runner = runner.replace('if checkpoint.exists():', 'if USE_CHECKPOINTS and checkpoint.exists():')
resources = '''# RESOURCE PROFILE — run this cell first, in a fresh kernel.
import os

RUN_PROFILE = 'local'  # 'local' for this Mac; 'professor' for a four-core cold run
LOCAL_CPU_THREADS = 12  # Change to 8 if the Mac becomes sluggish; speed is not benchmarked.

if RUN_PROFILE not in ('local', 'professor'):
    raise ValueError("RUN_PROFILE must be 'local' or 'professor'")
THREADS = min(LOCAL_CPU_THREADS if RUN_PROFILE == 'local' else 4, os.cpu_count() or 1)
if THREADS < 1:
    raise ValueError('CPU thread count must be positive')
# Professor timing must include actual fitting, not restored predictions.
USE_CHECKPOINTS = RUN_PROFILE == 'local'
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['OMP_NUM_THREADS'] = str(THREADS)
print(f'Profile: {RUN_PROFILE} | {THREADS} threads per model | one model at a time')
print('Restore completed tasks:', USE_CHECKPOINTS)
'''
previous = json.loads((ROOT/'notebooks/16_hydro_dispatch_challenger.ipynb').read_text())
config = ''.join(next(c['source'] for c in previous['cells'] if c.get('id')=='config')).split('SPECS =')[0]
config = re.sub(r"THREADS = min\(4,.*?os.environ\['OMP_NUM_THREADS'\] = str\(THREADS\)\n", '', config, flags=re.S)
config = re.sub(r"SOURCE_DIGEST = '[^']*'", f"SOURCE_DIGEST = '{digest}'", config)
config += '''SPECS = {}
for family, backend, parameters, cap in [
    ('xgb', 'xgboost', xgb_parameters, 2000),
    ('lgb', 'lightgbm', lgb_parameters, 1800),
]:
    for variant in ('physics', 'competition', 'competition_timing'):
        name = family + ('_v2' if variant=='physics' else '_' + variant)
        SPECS[name] = dict(backend=backend, features=variant, max_rounds=cap, parameters=dict(parameters))
print(f'CPU threads: {THREADS}. Sequential fits; notebook 12/13 settings and rounds are frozen.')
'''

build = '''data = build_project_data(ROOT)
extra, local, mapping = build_research_features(data)
additions, feature_summary = build_refinement_features(data)
runner = ChallengerRunner(data, extra, local, additions, SPECS, THREADS, SOURCE_DIGEST)
print('Cases:', data.n_train_cases, 'training;', data.n_test_cases, 'test')
display(feature_summary)
feature_summary.to_csv(runner.directory/'feature_groups.csv',index=False)
'''

validation = '''# TRAINING A: fixed-parameter feature ablation, then historical audit.
selection, scores = screen_refinements(runner, BASELINE_ROUNDS)
display(scores.pivot(index='model', columns='year', values='micro_auc'))
print('Chosen new-model pair:', selection['weights'])
print('Improved the 2022 audit:', selection['passed_audit'])
'''

final = '''# TRAINING B: exactly two selected new models per generator, trained on all history.
# Restart: run setup/definition/build cells, then this cell to resume final fitting.
selection = json.loads((runner.directory/'selection.json').read_text())
if (selection['source_digest'] != SOURCE_DIGEST or selection['raw_digest'] != runner.raw
        or selection['specs'] != SPECS):
    raise ValueError('Saved selection is for a different configuration. Rerun TRAINING A.')
train = np.arange(data.n_train_cases)
test = np.arange(data.n_train_cases, data.n_all_cases)
blend = np.zeros((data.n_test_cases,len(data.units),HORIZON),np.float32)
for member, weight in selection['weights'].items():
    predictions, _ = runner.run_stage(member, train, test, 'final', selection['rounds'][member])
    family = 'xgb' if SPECS[member]['backend']=='xgboost' else 'lgbm'
    path, _ = make_submission(data, predictions, f'{family}_physics_features_v4.csv')
    print('Single-model CSV:',path)
    blend += weight * predictions
submission_path, submission = make_submission(data,blend,'physics_features_v4_blend.csv')
print('NEW 50/50 BLEND READY:',submission_path)
print('Improved historical audit:',selection['passed_audit'])
if not selection['passed_audit']:
    print('This is still the NEW-model blend, not a baseline fallback. Historical results do not support an improvement.')
print('Total minutes:',round((time.perf_counter()-NOTEBOOK_STARTED)/60,1))
display(submission.iloc[:3,:8])
'''

export = '''# Freeze the selected pair into a self-contained final-only reproduction notebook.
original = ROOT/'notebooks/17_physics_feature_refinement.ipynb'
document = json.loads(original.read_text(encoding='utf-8'))
cells = []
for c in document['cells']:
    if c.get('id') in ('resources','config','pipeline','physics','runner','features','build'):
        c = json.loads(json.dumps(c))
        c['execution_count'],c['outputs'] = None,[]
        if c['id']=='resources':
            c['source'] = ''.join(c['source']).replace("RUN_PROFILE = 'local'", "RUN_PROFILE = 'professor'").splitlines(keepends=True)
        if c['id']=='features':
            c['source'] = ''.join(c['source']).split('def screen_refinements(')[0].splitlines(keepends=True)
        cells.append(c)
frozen = 'selection = '+repr(selection)+'\\n'
fitting = ('prediction = runner.final_predictions(selection)\\n'
           'path, submission = make_submission(data, prediction, "physics_features_v4_blend.csv")\\n'
           'print("Saved:",path)\\n'
           'print("Elapsed minutes:",(time.perf_counter()-NOTEBOOK_STARTED)/60)\\n')
for ident,source in [('frozen',frozen),('final',fitting)]:
    cells.append(dict(cell_type='code',id=ident,metadata={},execution_count=None,outputs=[],source=source.splitlines(keepends=True)))
intro = ('# Short notebook — physics feature refinement blend\\n\\n'
         '**Henrik Remmen (541747), Group 119, [119] SINTEF Databreach**\\n\\n'
         'Offline raw-data reproduction with four CPU threads. Models and rounds match '
         'notebooks 12/13; feature variants selected on 2021, audited on 2022 in notebook 17. '
         'Run all cells. No pretrained artifacts or other notebooks are required.\\n')
cells.insert(0,dict(cell_type='markdown',id='intro',metadata={},source=intro.splitlines(keepends=True)))
document['cells']=cells
short_path = runner.directory/'Short_notebook_physics_features_v4.ipynb'
short_path.write_text(json.dumps(document,indent=1,ensure_ascii=False)+'\\n',encoding='utf-8')
print('Final-only notebook:',short_path)
'''

intro = '''# 17 — Refine notebook 12/13 features and blend their predictions

**Henrik Remmen (541747), Group 119 — [119] SINTEF Databreach**

The working baseline is the physics XGBoost/LightGBM blend, public score 0.98990.
Notebook 16's new blend scored 0.98643 and changed both features and model capacity.
This experiment holds learning rate, depth/leaves, regularization, seeds and per-unit
tree counts fixed at the successful 12/13 settings. Only the feature set changes.

New family A: fraction of better hours and average price premium before/after each hour,
over 24, 72 and 168 hours. Family B: timing of inflow relative to profitable hours,
inflow-weighted prices, forthcoming reservoir floors and distance to remaining price peaks.
Test A first, then A+B. The full supplied weekly price/inflow paths are legitimate case
inputs. No observed storage after case start, commitment labels, fitted hydraulic simulator,
external data or solver outputs are used in feature engineering.

Run all cells in a fresh kernel. TRAINING A compares original, A and A+B on 2021 for
each model, picks the best NEW variant per backend, and audits the fixed 50/50 pair on 2022.
These historical years were used before, so the audit is not a pristine holdout. Fixed baseline
counts were also chosen in earlier development. TRAINING B always exports the new pair and
its blend, even if the audit is worse; it clearly reports whether improvement is supported.
The last cell exports a short notebook that trains only those two models from raw data.

The first code cell selects local (default 12 CPU threads) or professor (max four) mode.
Fits remain sequential to control memory. Float32 matrices; resumable local checkpoints.
Changing thread count reuses completed local tasks; small floating-point differences across
thread counts are possible. Professor mode disables cache reads for honest cold-run timing.
The exported short notebook ALWAYS starts in professor mode, regardless of the local setting.
Development: 140 unit fits. Final reproduction: 28 fits. Allow hours for development;
runtime and peak RAM during actual fitting are not yet measured. The professor's final-only
notebook has no feature search or validation runs and must be cold-timed against 12 hours/32 GB.
All required code is embedded. Libraries are listed in the project requirements.txt.
No model training was performed when this notebook was constructed. No score is guaranteed.
'''
cells = [cell('markdown',intro,'intro'),cell('code',resources,'resources'),cell('code',config,'config')]
for ident,source in [('pipeline',pipeline),('physics',physics),('runner',runner),('features',features)]:
    cells += [cell('markdown','## '+ident+' — definitions only',ident+'-heading'),cell('code',source,ident)]
for ident,heading,source in [('build','Build feature groups — no training',build),
                            ('validation','TRAINING A — controlled feature comparison',validation),
                            ('final','TRAINING B — new models and submission CSVs',final),
                            ('export','Export a self-contained short notebook',export)]:
    cells += [cell('markdown','## '+heading,ident+'-heading'),cell('code',source,ident)]
cells += [cell('markdown','''## Outputs and research rationale

Upload candidate: `submissions/physics_features_v4_blend.csv`.
Individual candidates: `submissions/xgb_physics_features_v4.csv` and `submissions/lgbm_physics_features_v4.csv`.
Diagnostics and short notebook: `artifacts/physics_features_v4/`.

Why these hypotheses: the existing features encode weekly rank, mean prices and price-minus-water
value, but do not directly describe competing higher-price hours separately before and after
the current hour. Existing cumulative inflow also lacks explicit alignment with profitable hours.
The new variables quantify these relationships without fitting another model or pretending
to reproduce the full hydraulic solver. They may help or hurt; the ablation measures this.

[SINTEF SHOP water values](https://docs.shop.sintef.energy/examples/water_values/water_values.html)
explain the opportunity cost of stored water and the interpretation of local EUR/MWh values.
[Scikit-learn temporal validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
supports testing on later observations; here 168-hour case overlaps are additionally purged.
These sources motivate the design, not a claimed improvement. Public scores never enter selection.
''','outputs')]
nb=dict(cells=cells,metadata=dict(kernelspec=dict(name='python3',language='python',display_name='Python 3'),
                                 language_info=dict(name='python')),nbformat=4,nbformat_minor=5)
path=ROOT/'notebooks/17_physics_feature_refinement.ipynb'
path.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print(path)
