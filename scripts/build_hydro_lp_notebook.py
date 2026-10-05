"""Serialize notebook 19 and its professor short notebook; no data processing or model training."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = 'artifacts/hydro_lp_v6'


def cell(kind, source, ident):
    result = dict(cell_type=kind, id=ident, metadata={}, source=source.strip().splitlines(keepends=True))
    if kind == 'code':
        ast.parse(source.strip())
        result.update(execution_count=None, outputs=[])
    return result


def embed(name):
    return '\n'.join(line for line in (ROOT / name).read_text().splitlines() if not line.startswith('from src.'))


experiment = embed('src/sintef_lp_experiment.py')
sources = {
    'pipeline': embed('src/sintef_pipeline.py'),
    'physics': embed('src/sintef_research_features.py'),
    'fit_helpers': embed('src/sintef_challenger_runner.py').split('class ChallengerRunner:')[0],
    'competition': embed('src/sintef_feature_refinement.py').split('def screen_refinements(')[0],
    'paths': embed('src/sintef_path_features.py'),
    'staged': embed('src/sintef_path_runner.py').split('class PathRunner')[0],
    'hydro_lp': embed('src/sintef_hydro_lp.py'),
    'experiment': experiment,
}
short_experiment = experiment.split('def group_diagnostics(')[0] + 'def verify_submission(' + experiment.split('def verify_submission(', 1)[1]
digest = hashlib.sha256(''.join(sources.values()).encode()).hexdigest()

resources = '''# FIRST CODE CELL: execution resources and experiment years. Statistical settings live in config.
import os
RUN_PROFILE = 'local'      # 'professor': max 4 CPU threads and no cache reads (cold run)
LOCAL_CPU_THREADS = 8      # 8 matches the thread count of the recorded notebook-18 runs
DEV_YEARS = (2020, 2021)   # model selection (pre-registered rule, see markdown above)
AUDIT_YEAR = 2022          # later check, evaluated once after selection.json exists
TRAIN_FINAL = True         # False: stop after the printed verdict, before the 28 final fits
if RUN_PROFILE not in ('local', 'professor'):
    raise ValueError('Unknown profile')
THREADS = min(LOCAL_CPU_THREADS if RUN_PROFILE == 'local' else 4, os.cpu_count() or 1)
USE_CHECKPOINTS = RUN_PROFILE == 'local'
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['OMP_NUM_THREADS'] = str(THREADS)
print(f'{RUN_PROFILE}: {THREADS} CPU threads, one model at a time, resume={USE_CHECKPOINTS}')
'''
config = f'''from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from IPython.display import display

NOTEBOOK_STARTED = time.perf_counter()
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'data/kernel').is_dir() and (p / 'prediction_mapping.csv').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Place the notebook beside the supplied data/ directory and mapping files.')
SOURCE_DIGEST = '{digest}'  # change if you edit any definition cell (invalidates caches)
ARTIFACTS = ROOT / '{ARTIFACTS}'

# Identical to multiscale_boosting_v5 (Kaggle public 0.99036): inherited per-unit counts x 2.
BASELINE_ROUNDS = {{
 'xgb': dict(zip(
  ['Hogga_G1','Tokke_G1','Tokke_G2','Tokke_G3','Tokke_G4','Lio_G1','Byrte_G1',
   'Vinje_G1','Vinje_G2','Vinje_G3','Haukeli_G1','Songa_G1','Kjela_G1','Vesle Kjela_G1'],
  [410,450,854,587,693,249,292,658,404,607,223,510,326,372])),
 'lgb': dict(zip(
  ['Hogga_G1','Tokke_G1','Tokke_G2','Tokke_G3','Tokke_G4','Lio_G1','Byrte_G1',
   'Vinje_G1','Vinje_G2','Vinje_G3','Haukeli_G1','Songa_G1','Kjela_G1','Vesle Kjela_G1'],
  [398,534,387,458,410,386,441,852,299,873,330,465,393,177]))}}
xgb_parameters = dict(objective='binary:logistic', eval_metric='auc', learning_rate=.03,
    max_depth=7, min_child_weight=50, subsample=.85, colsample_bytree=.85,
    reg_alpha=.1, reg_lambda=5., tree_method='hist', max_bin=256, device='cpu', random_state=42)
lgb_parameters = dict(objective='binary', metric='None', learning_rate=.03, num_leaves=31,
    max_depth=-1, min_data_in_leaf=150, feature_fraction=.85, bagging_fraction=.85,
    bagging_freq=1, lambda_l1=.1, lambda_l2=5., max_bin=255, device_type='cpu',
    deterministic=True, force_col_wise=True, seed=42, verbosity=-1)
ROUNDS = BASELINE_ROUNDS
SPECS = {{
    'xgb': dict(backend='xgboost', features='competition', max_rounds=2000, parameters=xgb_parameters),
    'lgb': dict(backend='lightgbm', features='competition', max_rounds=1800, parameters=lgb_parameters),
}}
'''
build = '''# Input-only feature construction, including one LP per case. No estimator is fitted here.
data = build_project_data(ROOT)
extra, local, mapping = build_research_features(data)
additions, _ = build_refinement_features(data)
competition = additions['competition']
del additions  # the rejected competition+timing variant is not used
paths, path_audit = build_path_features(data, local)
ARTIFACTS.mkdir(parents=True, exist_ok=True)
starts = pd.concat([data.train_cases, data.test_cases])['starttime']
lp_cache = ARTIFACTS / f'lp_cube_{SOURCE_DIGEST[:12]}_{raw_digest(ROOT)[:12]}.npz'
if USE_CHECKPOINTS and lp_cache.exists():
    with np.load(lp_cache) as saved:
        lp_cube, lp_plants, lp_seconds = saved['cube'], list(saved['plants']), float(saved['seconds'])
    print('Restored LP solutions:', lp_cache.name)
else:
    lp_cube, lp_plants, lp_seconds = solve_all_cases(ROOT, starts, THREADS)
    np.savez_compressed(lp_cache, cube=lp_cube, plants=np.array(lp_plants), seconds=lp_seconds)
print(f'LP relaxation: {len(lp_cube)} cases solved in {lp_seconds:.0f}s with {THREADS} threads')
n_cases = data.n_all_cases
price = data.X_all['price'].to_numpy().reshape(n_cases, HORIZON)
wv_source = {p: data.X_all[f'water_value_end__{PLANT_SOURCE[p]}'].to_numpy().reshape(n_cases, HORIZON)[:, 0]
             for p in lp_plants}
units_per_plant = {p: sum(u.rsplit('_G', 1)[0] == p for u in data.units) for p in lp_plants}
lp = lp_plant_features(lp_cube, lp_plants, price, wv_source, units_per_plant)
runner = LPRunner(data, extra, local, competition, paths, lp, SPECS, ROUNDS, THREADS,
                  SOURCE_DIGEST, ARTIFACTS, USE_CHECKPOINTS)
for variant in VARIANTS:
    print(variant, 'features per unit:', {p: len(runner.feature_names(p, variant)) for p in ('Tokke', 'Lio')})
'''
physics_check = '''# NO TRAINING. Physics sanity check: how well does ONE untrained score rank each unit's hours?
# rule = price - terminal local water value of the source reservoir; LP = reduced-cost margin.
# Labels are used descriptively only; nothing here feeds selection.
train_years = pd.DatetimeIndex(data.train_cases.starttime).year.to_numpy()
N = data.n_train_cases
rows = []
for i, unit in enumerate(data.units):
    plant = unit.rsplit('_G', 1)[0]
    j = lp_plants.index(plant)
    rule = (price - wv_source[plant][:, None])[:N]
    lp_score = lp_cube[:N, j, 2] + 1e-3 * (price[:N] - lp_cube[:N, j, 1])  # tie-break inside zero reduced cost
    for year in range(2015, 2023):
        s = train_years == year
        y = data.target_cube[s, i]
        rows.append(dict(unit=unit, year=year, rule_auc=roc_auc_score(y.ravel(), rule[s].ravel()),
                         lp_auc=roc_auc_score(y.ravel(), lp_score[s].ravel()),
                         lp_week_level_corr=np.corrcoef(lp_cube[:N][s, j, 0].mean(1), y.mean(1))[0, 1]))
physics = pd.DataFrame(rows)
physics.to_csv(ARTIFACTS / 'physics_check_untrained.csv', index=False)
display(physics.groupby('unit')[['rule_auc', 'lp_auc', 'lp_week_level_corr']].mean().round(4))
'''
shift = '''# NO TRAINING, NO LABELS: input distribution by case-start year, including the 2023-2024 test years.
years = pd.DatetimeIndex(starts).year.to_numpy()
rows = []
for year in sorted(set(years)):
    s = years == year
    row = dict(year=year, price_mean=price[s].mean(), price_std_within_week=price[s].std(axis=1).mean())
    for p in ('Tokke', 'Vinje', 'Lio', 'Haukeli', 'Hogga'):
        j = lp_plants.index(p)
        row[f'{p}_wv_mean'] = wv_source[p][s].mean()
        row[f'{p}_lp_week_fraction'] = lp_cube[s, j, 0].mean()
        # Hours the LP runs the plant although price is below its own water value (coupling-forced).
        row[f'{p}_lp_forced_share'] = ((lp_cube[s, j, 0] > .01) & (price[s] < wv_source[p][s][:, None])).mean()
    rows.append(row)
shift_table = pd.DataFrame(rows).set_index('year')
shift_table.to_csv(ARTIFACTS / 'input_shift_by_year.csv')
display(shift_table.round(3))
'''
development = '''# TRAINING A - model selection years. 4 fits x 14 units per year; completed units resume.
for year in DEV_YEARS:
    evaluate_year(runner, year, 'development')
log = pd.read_csv(ARTIFACTS / 'results_log.csv').drop_duplicates(['stage', 'year', 'model', 'factor'], keep='last')
display(log.query("stage == 'development'").pivot_table(index=['model', 'factor'], columns='year', values='micro_auc').round(6))
'''
selection_cell = '''# NO TRAINING: apply the pre-registered rule to the development years only.
selection = select_candidate(runner, DEV_YEARS)
print(json.dumps({k: v for k, v in selection.items() if k != 'identity'}, indent=1))
units = pd.concat([pd.read_csv(ARTIFACTS / f'diagnostics_development_{y}.csv') for y in DEV_YEARS])
display(units.query("group == 'unit' and model in ['control_v5', 'candidate_v5_lp']")
        .pivot_table(index='label', columns=['year', 'model'], values=['auc', 'weekly_level_mae']).round(4))
'''
audit = '''# TRAINING B - later check on AUDIT_YEAR, run once after selection. Do not retune on it.
selection = json.loads((ARTIFACTS / 'selection.json').read_text())
if selection['identity'] != runner.identity():
    raise ValueError('Configuration changed after selection; rerun development')
audit_summary, audit_table, audit_diag = evaluate_year(runner, AUDIT_YEAR, 'audit')
display(audit_table.pivot_table(index='model', columns='factor', values='micro_auc').round(6))
for group in ('unit', 'start_month', 'horizon_hours', 'utc_hour'):
    display(audit_diag.query("group == @group and model in ['control_v5', 'candidate_v5_lp']")
            .pivot_table(index='label', columns='model', values=['auc', 'weekly_level_mae'] if group == 'unit' else 'auc').round(4))
old = ROOT / 'artifacts/multiscale_boosting_v5/audit_predictions.npz'
if old.exists():  # consistency check: our control should reproduce notebook 18's 2022 candidate
    with np.load(old) as saved:
        mine = np.load(ARTIFACTS / f'predictions_audit_{AUDIT_YEAR}.npz')
        if AUDIT_YEAR == 2022 and np.array_equal(saved['case_ids'], mine['case_ids']):
            print('max |control - notebook18 v5| =', float(np.abs(saved['candidate'] - mine['control_v5']).max()))
'''
final = '''# TRAINING C - candidate pair on all labelled cases (2 x 14 fits) and the Kaggle CSVs.
# Refuses to run unless selection and later check were produced by this exact setup.
selection = load_checked(runner, 'selection.json')
audit_summary = load_checked(runner, f'summary_audit_{AUDIT_YEAR}.json')
label, message = verdict(selection, audit_summary)
print('Verdict for this setup:', label)
if TRAIN_FINAL:
    submission_path, submission = final_lp(runner)
    save_json(ARTIFACTS / 'final_manifest.json', dict(
        csv=str(submission_path.relative_to(ROOT)), verdict=label, message=message,
        development_deltas=selection['deltas'], audit_delta=audit_summary['delta'],
        audit_interval_p05_p50_p95=audit_summary['bootstrap_delta_p05_p50_p95'], identity=runner.identity()))
    display(submission.iloc[:3, :6])
else:
    print('TRAIN_FINAL = False: stopped before the final fits.')
print('Total minutes in this kernel:', round((time.perf_counter() - NOTEBOOK_STARTED) / 60, 1))
'''
short_final = '''# FINAL - LP candidate pair on all labelled cases (2 x 14 fits) and the Kaggle CSVs.
submission_path, submission = final_lp(runner)
display(submission.iloc[:3, :6])
print('Total minutes:', round((time.perf_counter() - NOTEBOOK_STARTED) / 60, 1))
'''
conclusion = '''# NO TRAINING: verdict from the saved summaries, only if they match this exact setup.
selection = load_checked(runner, 'selection.json')
audit_summary = load_checked(runner, f'summary_audit_{AUDIT_YEAR}.json')
lo, mid, hi = audit_summary['bootstrap_delta_p05_p50_p95']
print('Development deltas:', dict(zip(selection['development_years'], np.round(selection['deltas'], 6))),
      '| rule:', selection['rule'], '| accepted:', selection['accepted'])
print(f"{AUDIT_YEAR}: control {audit_summary['control']:.6f}, candidate {audit_summary['candidate']:.6f}, "
      f"delta {audit_summary['delta']:+.6f}, 90% block bootstrap {lo:+.6f} .. {hi:+.6f}, "
      f"share of resamples > 0: {audit_summary['bootstrap_share_positive']:.2f}")
label, message = verdict(selection, audit_summary)
print(label + ':', message)
print('2020-2022 all informed earlier work, and 2022 informed this hypothesis: none is an untouched holdout.')
'''
intro = '''# 19 — LP-relaksasjon av hele vassdraget som input til v5-boosting

**Henrik Remmen (541747), Gruppe 119 — [119] SINTEF Databreach**

Beste public-score nå: `multiscale_boosting_v5.csv` **0.99036**. Rapportert leder: 0.99562.
Denne notebooken er ett nytt, kontrollert forsøk. Den lover ingen score.

## Hypotese (presis)

*H1:* Å legge til per-stasjon-features fra en ukentlig **lineær relaksasjon (LP) av hele
Tokke–Vinje-nettet** (planlagt vannføring, redusert-kost-margin, dual-basert terskel og
planlagt magasinbane) til **nøyaktig** v5-modellen øker renset (purged) forward micro-ROC-AUC.
Forventet mekanisme (del av hypotesen, ikke vist): færre feil i ukentlig driftsnivå for
de hydraulisk koblede enhetene (Vinje, Tokke, Hogga, Lio, Haukeli). Alt annet er likt: samme 334
v5-features, XGBoost/LightGBM-parametere, seeds, 2× arvede trærantall og 50/50-blanding.

## Lokalt grunnlag (analyser gjort før notebooken ble bygd, uten ny boosting)

- **Nivåfeil er viktige, men ikke hele bildet.** Et label-informert orakel som kjenner hver
  case/enhets sanne ukentlige ON-andel gir AUC 0.9974/0.9969 (2021/2022) mot v5 0.9851/0.9868.
  Det er et tak, ikke bevis på riktig timing: orakelet får alle konstante case/enheter gratis
  (47 % i 2021, 39 % i 2022). Innen uker med både ON og OFF er v5s AUC 0.958/0.968 (vektet
  per positiv/negativ-par), så timingfeil betyr fortsatt noe.
- **Regelen pris > vannverdi** (kildemagasinets lokale EUR/MWh) er nesten perfekt for Byrte,
  Songa, Kjela og Vesle Kjela, men svak for koblede enheter (snitt 2019–22): Vinje G1 0.756,
  Haukeli 0.774, Lio 0.871, Tokke 0.915–0.929, Hogga 0.934.
- **Én utrent LP-feature** gir samme mål: Vinje G1 0.978, Haukeli 0.933, Lio 0.949,
  Tokke 0.981–0.986, Hogga 0.968. I 2022 er LP alene på høyde med v5 for Tokke G1/Vinje G1.
- **Lio** kjører under vannverdi nesten bare når minstevannføringen i r_Vest_Vassdraget ikke
  dekkes av naturlig tilsig (ON 43–93 % vs 1–28 %); LP-en fanger dette via dualverdien.
- **LP alene løser ikke nivået:** ukenivå-MAE 2022 er lavere enn v5 for bare 8 av 14 enheter,
  og klart høyere for Lio (0.194 vs 0.114) og Vinje G3 (0.104 vs 0.051). **Som tillegg hjelper
  den:** en enkel lineær nivåmodell tilpasset 2021 og målt 2022 går fra MAE 0.0576 (bare
  v5-nivå) til 0.0490 (v5 + LP-nivå), bedre for 12 av 14 enheter (ikke Lio og Songa).
- Liten stacker på lagrede out-of-sample-prediksjoner (tilpasset 2021, målt 2022):
  kontroll uten LP 0.97821, med LP 0.98612 (+0.0079). Én-års-stacking skader kalibrering
  mellom enheter (begge under v5 0.98682), så **den reelle testen er denne notebooken**.

## Validering og kontroll

- Utvidende tidsvinduer med purge: trening kun på case der hele 168-t-horisonten slutter før
  året starter. Tilstøtende daglige case deler 144 timer.
- **Modellvalg:** 2020 og 2021. Forhåndsregistrert regel: blend-delta i snitt > 0 og intet år
  dårligere enn −0.0005. Ingen andre valg: ingen vektsøk, ingen trærantallvalg
  (1×/1.5× lagres kun som diagnose).
- **Etterfølgende kontroll:** 2022, kjøres én gang etter at `selection.json` finnes.
- **Ærlighet:** 2020–2022 er brukt i tidligere arbeid, og 2022 ble brukt i proben over.
  Ingen av dem er urørte holdout-år. Blokk-bootstrap (7-dagers blokker) gir usikkerhet.
- **Konklusjon før slutttrening**, bare fra resultater med samme data/kode/parametere:
  `SUPPORTED` (valgt og 90 %-intervall i kontrollåret over 0), `POSITIVE BUT UNCERTAIN`
  (positivt punktestimat, intervallet når 0), `NOT CONFIRMED` eller `NOT SUPPORTED`.
  Sluttcellen nekter å kjøre mot valideringer fra et annet oppsett og skriver
  `final_manifest.json` som knytter CSV-en til verdikten. `TRAIN_FINAL = False` stopper før
  de 28 slutt-fitene.
- Logges: micro-AUC per år/modell/trærprefiks, AUC/Brier per enhet, startmåned,
  horisontblokk og UTC-time, ukenivå-MAE, kjøretid per enhet (`runtime_log.csv`).

## Ikke brukt

Ingen eksterne data, skjulte testetiketter, observerte volumer etter case-start, SHOP-resultater,
`total_calculation_time` eller Run No. LP-en er vår egen funksjon av tillatte inputs
(pris, tilsig, startvolum, sluttvannverdi, min-vannføring/-volum, topologi).

## Kjøring og ressurser

Kjør alle celler i en fersk kernel. Første kodecelle styrer tråder og år.
Fullførte enheter gjenopptas fra disk; en avbrutt enkelttrening starter på nytt.
Notebook-18-logger (8 tråder, 2× trær): ett 14-enheters sett tar ca. 9–11 min (XGB) og 12–16 min (LGB).
Utvikling (2 år × 4 sett) + kontroll (4 sett) + final (2 sett) ≈ 14 sett ≈ **2,5–3 t**.
LP for 3 647 case: ca. 1–3 min. Kortnotebooken (4 tråder, 28 fits) er anslått til 1–2 t, **ikke målt**.
'''
outputs = '''## Filer og hva du gjør etterpå

- `submissions/hydro_lp_v6.csv` (blend), `submissions/xgb_hydro_lp_v6.csv`, `submissions/lgb_hydro_lp_v6.csv`.
- `artifacts/hydro_lp_v6/`: `results_log.csv`, `runtime_log.csv`, `summary_*.json`, `diagnostics_*.csv`,
  `predictions_*.npz`, `selection.json`, `physics_check_untrained.csv`, `input_shift_by_year.csv`,
  importanser per enhet, LP-cache.
- Professor-notebook: `artifacts/hydro_lp_v6/Short_notebook_hydro_lp_v6.ipynb` (bygd av
  `scripts/build_hydro_lp_notebook.py`; selvstendig, 4 tråder, 28 fits, ingen cache).

`submissions/experiment_log.csv` endres ikke automatisk; legg inn en rad selv etter opplasting.
Last opp `hydro_lp_v6.csv` bare som et nytt forsøk; behold v5 som valgt fallback til private-
scoren er kjent. Offentlig leaderboard er én delmengde; bruk den varsomt.

Kilder: [SHOP reservoar](https://docs.shop.sintef.energy/objects/reservoir/reservoir.html),
[SHOP vannverdier](https://docs.shop.sintef.energy/examples/water_values/water_values.html),
[SHOP elv](https://docs.shop.sintef.energy/objects/river/river.html),
[SINTEF InterOpt](https://www.sintef.no/en/latest-news/2026/how-ai-could-squeeze-more-value-from-norwegian-hydropower/),
[Xavier, Qiu & Ahmed 2021](https://pubsonline.informs.org/doi/10.1287/ijoc.2020.0976),
[Gasse et al. 2019](https://arxiv.org/abs/1906.01629),
[SciPy HiGHS](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html),
[scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
Detaljer: `RESEARCH_HYDRO_LP_V6.md`.
'''
definitions = [(name, source) for name, source in sources.items()]
cells = [cell('markdown', intro, 'intro'), cell('code', resources, 'resources'), cell('code', config, 'config')]
for name, source in definitions:
    cells += [cell('markdown', f'## {name} — definitions only', f'{name}-heading'), cell('code', source, name)]
for ident, title, source in [
        ('build', 'Features og LP — ingen trening', build),
        ('physics_check', 'Fysikk-sjekk uten trening', physics_check),
        ('shift', 'Endring i pris, vannverdi og LP-drift per år (ingen labels)', shift),
        ('development', 'TRENING A — modellvalg (2020, 2021)', development),
        ('selection', 'Valg etter forhåndsregistrert regel — ingen trening', selection_cell),
        ('audit', 'TRENING B — etterfølgende kontroll (2022)', audit),
        ('conclusion', 'Konklusjon før slutttrening — ingen trening', conclusion),
        ('final', 'TRENING C — endelig Kaggle-CSV (sjekker samme oppsett som valideringen)', final)]:
    cells += [cell('markdown', f'## {title}', f'{ident}-heading'), cell('code', source, ident)]
cells.append(cell('markdown', outputs, 'outputs'))
meta = dict(kernelspec=dict(name='python3', language='python', display_name='Python 3'), language_info=dict(name='python'))
research_path = ROOT / 'notebooks/19_hydro_lp_relaxation.ipynb'
research_path.write_text(json.dumps(dict(cells=cells, metadata=meta, nbformat=4, nbformat_minor=5), indent=1, ensure_ascii=False) + '\n', encoding='utf-8')

short_intro = '''# Short notebook — v5 features + watercourse LP relaxation, XGBoost/LightGBM blend

**Henrik Remmen (541747), Group 119 — [119] SINTEF Databreach**

Reproduces `submissions/hydro_lp_v6.csv` from the supplied raw data only. Place this notebook
beside `data/`, `prediction_mapping.csv` and `sample_submission.csv`; install the pinned
`requirements.txt`; run all cells. CPU only, at most four threads, one model at a time,
no saved models/predictions/caches are read. Work: one LP per case (3,647 small linear programs,
SciPy/HiGHS dual simplex), then 2 x 14 per-unit boosted models. Fixed hyperparameters and tree
counts (inherited per-unit counts x 2) and the 50/50 blend come from the development notebooks
(notebooks 07/13 counts, notebook 18 multiplier, notebook 19 LP features); see the report.
Expected runtime about 1-2 hours on four cores (not yet measured on the grading machine).
'''
short_build = build.split("for variant in VARIANTS:")[0]
short_cells = [cell('markdown', short_intro, 'intro'),
               cell('code', '\n'.join(l for l in resources.replace("RUN_PROFILE = 'local'", "RUN_PROFILE = 'professor'").splitlines()
                                        if not l.startswith(('DEV_YEARS', 'AUDIT_YEAR', 'TRAIN_FINAL'))), 'resources'),
               cell('code', config, 'config')]
for name, source in definitions:
    source = short_experiment if name == 'experiment' else source
    short_cells.append(cell('code', source, name))
short_cells += [cell('code', short_build, 'build'),
                cell('code', short_final, 'final')]
short_path = ROOT / ARTIFACTS / 'Short_notebook_hydro_lp_v6.ipynb'
short_path.parent.mkdir(parents=True, exist_ok=True)
short_path.write_text(json.dumps(dict(cells=short_cells, metadata=meta, nbformat=4, nbformat_minor=5), indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(research_path)
print(short_path)
print('source digest', digest)
