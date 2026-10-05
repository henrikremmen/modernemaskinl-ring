# Dispatch challenger, 26 September 2026

## Evidence and objective

User-reported public scores: physics XGBoost 0.98979, physics LightGBM 0.98957,
plant-state LightGBM 0.98934, 50/50 physics blend 0.98990. Reported leader: 0.99562.
Closing this gap requires reducing 1-AUC by about 57%; minor weight tuning is
unlikely to suffice. This is a hypothesis-driven attempt, not a guaranteed score.

The original local pipeline, raw input definitions, task and grading PDFs, YAML
topology, available OOF predictions, model parameters and recorded runtimes were
inspected. No model was trained during this investigation.

Existing physics LightGBM OOF per-unit AUC (pooled valid historical cases):

| Unit | AUC | Brier error | Fraction of Brier error near a target transition |
|---|---:|---:|---:|
| Haukeli G1 | 0.87432 | 0.21052 | 0.039 |
| Vinje G2 | 0.94064 | 0.09986 | 0.235 |
| Lio G1 | 0.95124 | 0.09075 | 0.146 |
| Vinje G3 | 0.95615 | 0.07785 | 0.277 |
| Tokke G1 | 0.98987 | 0.03854 | 0.354 |

Near-transition means within two hours of a change in true commitment, within
each case. This diagnostic uses training labels only. Per-unit AUC/Brier are
descriptive and are not a decomposition of the global Kaggle micro AUC.
Haukeli's errors are predominantly persistent state errors, making smoothing
alone an unconvincing solution. It has a small direct reservoir and upstream
hydraulic dependencies. Higher model capacity and storage-budget features are
plausible alternatives; neither has been validated yet.

Physics improved LightGBM's 2020 score from 0.917695 to 0.948565 and its 2021
score from 0.975438 to 0.983362, but 2022 fell from 0.978565 to 0.976781.
This variation argues for evaluating whole future years. Existing models already
have high test prediction correlation; simply accumulating more similar models
does not establish an improvement.

## What notebook 16 tests

1. Extend the local reservoir features with upstream reservoirs connected through
   tunnels (for example Vaamarvatn/Totak/Venemo and Foersvatn/Bordalsvatn).
   These are approximate availability groups, not instantaneous fully mixed pools.
2. Express initial water and inflow as possible hours at rated plant output.
   Approximate discharge uses initial head and 90% efficiency, explicitly a proxy.
   Add scarcity price, overflow pressure, remaining supply, downstream fill,
   profit-window and daily price-shape features.
   A separate passive catchment traces rivers as well as tunnels, including
   Langeidvatn feeding Vatjern/Haukeli. Its totals are proxies, not assured release.
3. Compute a small two-state dynamic program over each supplied weekly price
   path. ON/OFF objective differences at several switching-cost scales describe
   sustained profitability. The initial and terminal states are free. The proxy
   does not solve the full hydraulic system and is used as an input to ML only.
4. Train an XGBoost depth-8 challenger and LightGBM 63-leaf challenger with the
   same rich features. Preserve two fixed physics controls. More capacity is a
   testable hypothesis, not an assumption that bigger is better.

No observed volume after case start, true companion-unit commitment, solver
runtime or outputs, hidden labels, external dataset, or external service is an
input. Terminal water values and all 168 hours of supplied price/inflow are valid
case inputs according to the course definitions. No SHOP installation is needed.

## Model selection and reproduction

- Purge 168-hour case overlap at each temporal split.
- 2021: fit through 2020, select new model rounds using early stopping. Old
  controls use their previously selected fixed counts.
- Choose among five global mixing weights of the old and new 50/50 ensembles.
- 2022: refit on pre-2022 history with frozen rounds; audit the chosen mixture.
  Fall back to the old mixture if it does not improve global micro AUC.
- Refit on all labeled data, export the guarded prediction and a separately named
  pure challenger. These are not the same artifact if the audit rejects it.
- Automatically export final-only short notebooks with frozen counts and weights,
  all code embedded, and raw-data training. Grading requires only two selected
  short notebooks overall, not every research notebook.

Development scores are optimistic because early stopping uses 2021. 2022 is a
historical audit, not a pristine holdout: previous features/hyperparameters already
used these years. Acceptance on it is another selection decision. The code never
uses public leaderboard scores to tune weights. Fixed global weights avoid a
large per-generator search and preserve cross-generator probability comparisons.

The baseline LightGBM uses only seed=42, matching its original saved model. Its
derived bagging and feature-fraction seeds are not both 42. Fixed tree counts
match the saved physics model, rather than the control feature model.

## Compute

Four CPU threads maximum, sequential fits, BLAS one thread, float32 matrices,
per-unit checkpoints fingerprinted by raw files, embedded source, parameters,
splits, versions and rounds. Interrupted individual fits restart; completed ones
resume. Development runs 112 unit fits plus at most 56 final fits. A selected
final-only notebook needs at most 56 fits, no validation. New fits cap at 1400
rounds with early stopping only in development. Existing four-thread XGBoost
final training took 459.5 seconds; plant-state training took 1045.9 seconds.
New runtime and peak memory are not measured. Allow hours for development;
benchmark each selected final notebook cold on four cores against the 12-hour,
32-GB grading target. This is a resource design, not a runtime guarantee.

## Primary sources

- [SINTEF SHOP water value descriptions](https://docs.shop.sintef.energy/examples/water_values/water_values.html).
  Local EUR/MWh water values refer to the first downstream plant. The proposed
  features keep this meaning and do not double-subtract downstream values.
- [SINTEF SHOP generator model](https://docs.shop.sintef.energy/objects/generator/generator.html).
  Head, discharge and efficiency determine power; start/stop costs affect UC.
- [LightGBM parameter tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html).
- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).
- [CatBoost original paper](https://papers.nips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).
  Considered as a diversity alternative, but not added to the first bounded
  experiment: existing CatBoost public score is lower, and the immediate feature
  deficiencies offer a more specific hypothesis.

Local sources: `Dataset_definitions_and_explanation.pdf`, `ML_Task_TDT4173.pdf`,
`TDT4173_Project_Submission_and_Grading.pdf`, supplied topology YAML and historical
prediction artifacts. Current web docs may postdate the dataset's SHOP version;
the new calculations are explicitly approximations, not a replica of that solver.
