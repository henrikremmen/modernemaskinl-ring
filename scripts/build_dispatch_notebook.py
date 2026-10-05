"""Build the self-contained notebook; this script never fits any model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, ident):
    value = dict(cell_type=kind, id=ident, metadata={}, source=source.strip().splitlines(keepends=True))
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def embedded(path):
    return '\n'.join(line for line in (ROOT / path).read_text().splitlines()
                     if not line.startswith('from src.'))


sources = [embedded(p) for p in ('src/sintef_pipeline.py', 'src/sintef_research_features.py',
                                 'src/sintef_challenger_features.py', 'src/sintef_challenger_runner.py')]
digest = hashlib.sha256('\n'.join(sources).encode()).hexdigest()
config = '''from pathlib import Path
import os
import time
from IPython.display import display

NOTEBOOK_STARTED = time.perf_counter()
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'data/kernel').is_dir() and (p / 'prediction_mapping.csv').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Place notebook beside the supplied data/ directory and mapping files.')
THREADS = min(4, max(1, os.cpu_count() or 1))
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['OMP_NUM_THREADS'] = str(THREADS)
SOURCE_DIGEST = '__DIGEST__'

# Existing models are frozen controls. Their counts were determined in prior development.
BASELINE_ROUNDS = {
 'xgb_v2': dict(zip(
  ['Hogga_G1','Tokke_G1','Tokke_G2','Tokke_G3','Tokke_G4','Lio_G1','Byrte_G1',
   'Vinje_G1','Vinje_G2','Vinje_G3','Haukeli_G1','Songa_G1','Kjela_G1','Vesle Kjela_G1'],
  [410,450,854,587,693,249,292,658,404,607,223,510,326,372])),
 'lgb_v2': dict(zip(
  ['Hogga_G1','Tokke_G1','Tokke_G2','Tokke_G3','Tokke_G4','Lio_G1','Byrte_G1',
   'Vinje_G1','Vinje_G2','Vinje_G3','Haukeli_G1','Songa_G1','Kjela_G1','Vesle Kjela_G1'],
  [398,534,387,458,410,386,441,852,299,873,330,465,393,177]))}
xgb_parameters = dict(objective='binary:logistic', eval_metric='auc', learning_rate=.03,
    max_depth=7, min_child_weight=50, subsample=.85, colsample_bytree=.85,
    reg_alpha=.1, reg_lambda=5., tree_method='hist', max_bin=256, device='cpu', random_state=42)
lgb_parameters = dict(objective='binary', metric='None', learning_rate=.03, num_leaves=31,
    max_depth=-1, min_data_in_leaf=150, feature_fraction=.85, bagging_fraction=.85,
    bagging_freq=1, lambda_l1=.1, lambda_l2=5., max_bin=255, device_type='cpu',
    deterministic=True, force_col_wise=True, seed=42, verbosity=-1)
SPECS = {
 'xgb_v2': dict(backend='xgboost', features='physics', max_rounds=2000, parameters=xgb_parameters),
 'lgb_v2': dict(backend='lightgbm', features='physics', max_rounds=1800, parameters=lgb_parameters),
 'xgb_dispatch': dict(backend='xgboost', features='dispatch', max_rounds=1400,
    parameters=dict(xgb_parameters, learning_rate=.04, max_depth=8, min_child_weight=20,
                    reg_lambda=8., colsample_bytree=.9)),
 'lgb_dispatch': dict(backend='lightgbm', features='dispatch', max_rounds=1400,
    parameters=dict(lgb_parameters, learning_rate=.04, num_leaves=63, min_data_in_leaf=100,
                    lambda_l2=8., feature_fraction=.9)),
}
print(f'At most {THREADS} CPU threads, one fit at a time. Training has not started.')
'''.replace('__DIGEST__', digest)

build = '''data = build_project_data(ROOT)
extra, local, original_mapping = build_research_features(data)
dispatch, storage_mapping = build_challenger_features(data)
runner = ChallengerRunner(data, extra, local, dispatch, SPECS, THREADS, SOURCE_DIGEST)
print('Cases:', data.n_train_cases, 'train;', data.n_test_cases, 'test')
print('Extra dispatch features per plant:', {p: len(v.columns) for p, v in dispatch.items()})
print('Resident feature arrays, excluding temporary training copies (GB):', round(
    (runner.base.nbytes + sum(v.memory_usage(index=False).sum() for v in local.values())
     + sum(v.memory_usage(index=False).sum() for v in dispatch.values())) / 1e9, 2))
display(storage_mapping)
storage_mapping.to_csv(runner.directory / 'storage_feature_mapping.csv', index=False)
'''

training = '''# TRAINING A — development and historical audit. Completed unit tasks resume.
selection, validation_scores = screen_challenger(runner, BASELINE_ROUNDS)
display(validation_scores.pivot(index='model', columns='year', values='micro_auc'))
print('Recommended final weights:', selection['weights'])
print('Historical audit accepted the challenger:', selection['passed_audit'])
'''

final = '''# TRAINING B — full-history fits, with already chosen rounds and weights.
# After a restart: rerun configuration/definitions/data, then load the saved selection.
if 'selection' not in globals():
    selection_path = runner.directory / 'selection.json'
    if not selection_path.exists():
        raise FileNotFoundError('Run TRAINING A before TRAINING B.')
    selection = json.loads(selection_path.read_text())
    if (selection.get('source_digest') != SOURCE_DIGEST or selection.get('raw_digest') != runner.raw
            or selection.get('specs') != SPECS):
        raise ValueError('Saved selection belongs to changed code/data. Rerun TRAINING A.')
guarded = runner.final_predictions(selection)
submission_path, submission = make_submission(data, guarded, 'hydro_dispatch_v3.csv')
print('Recommended CSV:', submission_path)
print('Passed historical audit:', selection['passed_audit'])
# Export a separately named pure challenger to make the experiment inspectable.
experimental_selection = dict(selection, weights={'xgb_dispatch': .5, 'lgb_dispatch': .5})
experimental = runner.final_predictions(experimental_selection)
experimental_path, _ = make_submission(data, experimental, 'hydro_dispatch_experimental.csv')
print('Pure challenger (inspect validation before upload):', experimental_path)
print(f'Total elapsed including features and validation: {(time.perf_counter()-NOTEBOOK_STARTED)/60:.1f} minutes')
display(submission.iloc[:3, :8])
'''

export = '''# Generate self-contained final-only notebooks with the chosen constants embedded.
# The generated notebooks do not read this notebook, src/, caches, or input submissions.
source_path = ROOT / 'notebooks/16_hydro_dispatch_challenger.ipynb'
if not source_path.is_file():
    raise FileNotFoundError('Keep notebook 16 under notebooks/ while exporting the short versions.')
source_notebook = json.loads(source_path.read_text(encoding='utf-8'))

def export_short(chosen, filename, output_csv):
    keep = []
    for c in source_notebook['cells']:
        if c['id'] in ('config', 'pipeline', 'physics', 'dispatch', 'runner', 'build'):
            copied = json.loads(json.dumps(c))
            copied['execution_count'] = None
            copied['outputs'] = []
            if c['id'] == 'runner':
                text = ''.join(copied['source']).split('def screen_challenger(')[0]
                copied['source'] = text.splitlines(keepends=True)
            keep.append(copied)
    constants = 'selection = ' + repr(chosen) + '\\n'
    execute = ('prediction = runner.final_predictions(selection)\\n'
               + f'path, submission = make_submission(data, prediction, {output_csv!r})\\n'
               + 'print("Saved:", path)\\n'
               + 'print("Elapsed minutes:", (time.perf_counter()-NOTEBOOK_STARTED)/60)\\n')
    # Only scalar hyperparameters, unit tree counts and blend weights are embedded.
    keep += [dict(cell_type='code', id='frozen-selection', metadata={}, execution_count=None, outputs=[], source=constants.splitlines(keepends=True)),
             dict(cell_type='code', id='final-fit', metadata={}, execution_count=None, outputs=[], source=execute.splitlines(keepends=True))]
    intro = ('# Short notebook — hydro dispatch ensemble\\n\\n'
             '**Henrik Remmen (541747), Group 119, [119] SINTEF Databreach**\\n\\n'
             'Offline raw-data reproduction; max four CPU threads. Run all cells. '
             'Counts/weights selected in notebook 16 and described in the research report. '
             'No precomputed predictions or trained models are required.\\n')
    keep.insert(0, dict(cell_type='markdown', id='short-intro', metadata={}, source=intro.splitlines(keepends=True)))
    result = dict(source_notebook, cells=keep)
    destination = runner.directory / filename
    destination.write_text(json.dumps(result, indent=1, ensure_ascii=False)+'\\n', encoding='utf-8')
    print('Final-only notebook:', destination)

export_short(selection, 'Short_notebook_dispatch_v3.ipynb', 'hydro_dispatch_v3.csv')
export_short(experimental_selection, 'Short_notebook_dispatch_experimental.ipynb', 'hydro_dispatch_experimental.csv')
'''

intro = '''# 16 — Hydro dispatch challenger: storage networks + start/stop economics

**Henrik Remmen (541747), Group 119 — [119] SINTEF Databreach**

Goal: improve on public 0.98990. The reported leader is 0.99562; no hidden-test score is guaranteed.
This notebook is an untrained experiment, not a measured improvement.

New features capture connected storage through tunnels, water budgets in full-load hours,
price cutoffs under those budgets, and a two-state dynamic program accounting for start/stop costs.
The approximate dispatch uses input data only and is not a SHOP reconstruction. Features
retain local water values in EUR/MWh; they do not subtract a downstream water value twice.
Two additional boosting models allow richer interactions. A 50/50 physics ensemble is the control.

Run all cells in a fresh kernel, one notebook at a time. Training A selects rounds and one
global blend weight on 2021, then checks it on 2022. Training B creates CSVs. The last cell
exports standalone final-only notebooks for the professor: no validation reruns, source files,
saved model files or old CSVs are needed. Dependencies are in requirements.txt; no downloads occur.

Resource budget: four CPU threads, sequential fits, float32 features, resumable unit checkpoints.
Development entails 112 unit fits and up to 56 final fits; early stopping bounds new-model work.
Final-only reproduction requires at most 56 unit fits. The observed earlier 14-model XGBoost
run took 7.7 minutes on this Mac at four threads; the larger new models are unbenchmarked.
Allow hours for the development run. The 12-hour requirement applies per final short notebook;
confirm cold-start runtime on four cores before submission. Peak RAM is not hard-capped.

All years have informed prior project work; 2022 is a historical audit, not a pristine holdout.
The old controls also use previously selected constants. The development score is optimistic
because it selects boosting rounds. Public Kaggle scores never enter the selection code.
If the proposed mixture fails the audit, the recommended CSV reverts to the baseline and a
separate experimental CSV remains available. The notebook reports this explicitly.
'''

cells = [cell('markdown', intro, 'intro'), cell('code', config, 'config')]
for ident, source in zip(('pipeline','physics','dispatch','runner'), sources):
    cells += [cell('markdown', '## Definitions: ' + ident + ' (no training)', ident+'-heading'),
              cell('code', source, ident)]
cells += [cell('markdown', '## Build features and inspect the storage mapping (no training)', 'build-heading'),
          cell('code', build, 'build'), cell('markdown', '## TRAINING A — historical validation', 'train-a-heading'),
          cell('code', training, 'train-a'), cell('markdown', '## TRAINING B — full-history predictions', 'train-b-heading'),
          cell('code', final, 'train-b'), cell('markdown', '## Export short notebooks', 'export-heading'),
          cell('code', export, 'export'), cell('markdown', '''## Sources and interpretation

- [SINTEF SHOP water values](https://docs.shop.sintef.energy/examples/water_values/water_values.html): local EUR/MWh values are relative to the first downstream plant; SHOP converts these internally.
- [SINTEF SHOP generators](https://docs.shop.sintef.energy/objects/generator/generator.html): head, discharge, efficiency and start/stop economics motivate the proxy features.
- [LightGBM tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html): leaves and minimum leaf size control capacity and cost.
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html): bounded histogram tree training and regularization.

These sources motivate hypotheses, not a prediction of Kaggle score. Only the supplied course data
are used. The features approximate hydraulic connectivity and do not enforce full hydraulic feasibility.
See `RESEARCH_DISPATCH_V3.md` for local error analysis and selection rationale.
''', 'sources')]
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                         language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
destination = ROOT / 'notebooks/16_hydro_dispatch_challenger.ipynb'
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
print(destination)
