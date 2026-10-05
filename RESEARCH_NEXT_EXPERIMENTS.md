# Next hydropower experiments — 25 September 2026

Scope: review of the current project data, metric, pipeline, EDA/feature notebooks,
model notebooks 03–11, saved predictions and experiment log, project PDFs and
relevant course slides; followed by primary-source research. New notebooks were
created but **not executed**. No model was fitted during this work.

## Findings from this project

The recorded public scores are 0.98825 for the temporal ensemble, 0.98824 for
XGBoost, 0.98774 for the greedy blend, 0.98716 for the logistic stack, 0.98549 for
CatBoost, 0.98061 for improved LightGBM and 0.92747 for TCN. A public difference
of 0.00001 does not establish a reliable improvement. There are no private test
labels available locally. Keep both strong candidates and use chronological
validation to guide further work.

The greedy blend uses 52.25% XGBoost, 42.75% CatBoost and 5% TCN. Its local mean
AUC is 0.971444 versus 0.968969 for XGBoost. Thus model diversity is useful even
when the contributing model is weak alone. Public ranking and local ranking
already disagree, which argues against leaderboard-driven hyperparameter search.

I recomputed per-unit AUC from the saved OOF cubes and labelled cases, without
fitting any model. XGBoost's mean annual unit AUC is lowest for Haukeli G1
(0.90036), Lio G1 (0.94964) and Vinje G3 (0.95736). Byrte G1 is already very
strong (0.99686). These unit scores are diagnostics, not the competition's
global micro AUC; improving the weakest unit does not automatically improve
the global ranking of all unit probabilities.

Important gaps in the implementation:

- Notebook 04 chooses a feature set using logistic regression, then passes that
  same set to LightGBM. The LightGBM result is not a dedicated full-feature
  LightGBM experiment. Notebooks 05 and 07 do use all 175 pipeline features.
- `src/sintef_pipeline.py` has reservoir filling, terminal water values,
  individual inflows and price windows. It lacks per-reservoir cumulative and
  remaining inflow volumes, local head proxies and within-day price ranks.
- It does not explicitly map each plant to its feeding reservoirs. The mapping
  from the supplied YAML includes Songa's two tunnel-fed reservoirs, and small
  Vatjern feeding Haukeli. Summing all reservoirs can obscure these distinctions.
- Notebook 11's logistic stack uses the base models' predictions for the same
  unit; it does not model the full joint configuration of generators in a plant.
- Tokke and Vinje show concentrated joint states, with occasional exceptions.
  These are suitable for testing a joint distribution, but not for imposing
  hard generator ordering or forcing same-plant predictions to be identical.
- Adjacent daily scheduling cases disagree on about **12.23%** of true states
  for the same overlapping hours, across all labelled cases and units. This
  was measured by comparing `y[:-1, :, 24:]` and `y[1:, :, :144]`. The same
  timestamp is not the same optimization problem: initial storage, terminal
  value and the remaining horizon change. Heavy overlap averaging is risky.
- EDA identifies major annual price/water-value shifts. A recent-years-only
  training window is not an obvious solution: 2024 is not simply another 2022.
- Existing boosting uses each outer validation year for early stopping. It
  therefore influences the chosen model. The new experiments isolate stopping
  in earlier history. Old scores remain useful references, but are not an exact
  controlled comparison with the new protocol.

`Team_10/` concerns an older raw-material receivals forecasting project with a
different metric. It is not evidence for this unit-commitment task.
`final_submission/` currently contains skeletons, not finished reproductions.

## Research and how it changes the experiment design

SINTEF describes local water values in EUR/MWh and conversion to global water
values using production/discharge factors and downstream contributions. This
supports plant-linked opportunity-cost features, but **not** treating the raw
difference between two reservoir EUR/MWh columns as an exact hydraulic cost.
The new features keep terminal values explicit and mark multi-source summaries
as approximations. [SHOP water values](https://docs.shop.sintef.energy/examples/water_values/water_values.html)

SHOP includes generator start and stop costs. This motivates features for
consecutive profitable hours and accumulated positive margins around the current
hour, rather than using only an instantaneous threshold. The rated-power
payback feature is a hypothesis: actual output, head losses and coupled dispatch
are unknown. [SHOP generators](https://docs.shop.sintef.energy/objects/generator/generator.html)

A broad tabular benchmark found strong tree-model performance and sensitivity
of neural models to feature properties. It is not a result on this dataset,
but together with our own model scores it supports prioritizing boosting before
another large neural architecture. [Grinsztajn et al., 2022](https://arxiv.org/abs/2207.08815)

LightGBM exposes CPU thread limits, deterministic histogram construction and
binary/multiclass objectives. These make controlled independent models and a
plant-state distribution practical experiments here. The new code sets explicit
thread budgets and column-wise histograms to limit memory overhead.
[LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html)

XGBoost's documented GPU path uses CUDA. Apple MPS is not that backend, so the
XGBoost experiment uses CPU histogram training.
[XGBoost GPU support](https://xgboost.readthedocs.io/en/stable/gpu/index.html)

## New notebooks and suggested run order

| Order | Notebook | Experiment | Assessment |
|---|---|---|---|
| 1 | `notebooks/13_lightgbm_physics_v2.ipynb` | Full 175-feature LightGBM versus the same model with physical features | Strongest untested gap; good first run |
| 2 | `notebooks/12_xgboost_physics_v2.ipynb` | Matched XGBoost control versus physical features | Builds directly on the strongest individual model |
| 3 | `notebooks/14_lightgbm_plant_states_v2.ipynb` | Joint state probabilities for each plant | More exploratory; inspect Tokke/Vinje and rare states |

The extended representation contains 268 features per task: 175 original,
75 global additions and 18 local plant additions. The additions cover:

- Current price rank within its 24-hour day; price relative to preceding and
  remaining horizon means; best remaining price gap; horizon day and time left.
- Per-reservoir cumulative/remaining inflow in million m3 relative to capacity.
- Storage slack and possible excess under a **no-release approximation**.
- Plant-specific fill, terminal water-value margin, normalized margin and initial
  gross head from the supplied volume/head curve and outlet height.
- Duration of positive-margin runs, 6-/24-hour positive and negative margin sums,
  and a rough start/stop-cost payback ratio.

No-release water budgets are not actual trajectories and omit upstream dispatch,
spill, tunnel flow and river travel times. They can exceed capacity and are not
clipped into a purported feasible state. Gross head also omits hydraulic losses.
The model must prove that these proxies help against the controls.

The plant-state experiment represents Tokke's four generators with 16 possible
states and Vinje's three with eight. It trains a multiclass distribution and
sums state probabilities to get each generator's ON probability. Single-unit
plants remain binary. Labels of neighbouring generators are used only to define
the training outcome, never as inference inputs. Rare or unseen combinations
may hurt, and multiple trees per boosting round can increase runtime.

## Validation, execution and restart behavior

All three notebooks are self-contained: their shared implementation is embedded
in notebook cells. They start from supplied raw data without requiring `src/`,
OOF predictions or saved models. The source modules are also present for review
and pure helper tests.

The training protocol is:

1. Outer years remain 2020, 2021 and 2022, with the existing 167-hour horizon-end
   purge before each year's first case.
2. The final 120 days within the outer training history are an inner stopping
   set. Its training portion is also purged.
3. Select tree count on that inner set, then refit using the complete outer
   training history and score the outer year.
4. For the final submission, use the median inner-selected count across folds
   and fit all labelled cases. Test labels are never involved.

The outer years have already informed many experiments, so they are not a new
untouched test set. Avoid large searches and interpret small differences cautiously.
Judge each year and unit alongside mean/worst micro AUC. As a screening heuristic,
require a mean improvement with worst-fold deterioration below about 0.001; this
is not a statistical significance test. Confirm a promising candidate with a
second seed or a prespecified additional historical block before strong claims.

On the current 18-core/24-GB Mac, defaults are **3 concurrent model fits × 4 CPU
threads**, with BLAS capped separately at one thread during the pool. Arrays
use float32; large feature matrices are shared between threads rather than
copied to child Python processes. Run one training notebook at a time. This is
a conservative resource budget, not a measured fastest setting or hard RAM cap.
Set `OUTER_JOBS=1` if memory pressure is high. For a four-core reproduction target,
set `CPU_BUDGET=4` before the derived settings.

Each finished fold/unit or fold/plant saves predictions, tree count and feature
importance. Inner stopping counts are saved before refitting. A restart restores
completed tasks; an interrupted individual fit restarts. Final native models are
also saved. A changed data/configuration/dependency/source snapshot creates a
separate checkpoint directory. Export protection requires a new experiment name
if an existing submission belongs to another signature.

Both feature-control notebooks contain two variants by default. With three outer
folds, each task involves an inner fit plus refit, followed by a final fit.
That is up to 196 fits for each two-variant, 14-unit notebook, and 63 for the
nine-plant notebook. Early stopping and caching reduce repeat work; runtime has
not been measured. Run the first notebook and assess its result before investing
in all experiments. A fit is not necessarily comparable in cost across objectives.

## Artifacts and later ensemble work

Outputs use new names:

- `xgb_control_v2.csv`, `xgb_physics_v2.csv`
- `lgbm_full_v2.csv`, `lgbm_physics_v2.csv`
- `lgbm_plant_states_v2.csv`

Each has a matching OOF NPZ, fold-score CSV and manifest under
`artifacts/predictions/`. Detailed per-unit diagnostics, feature importances and
checkpoints live in `artifacts/research_runs/`. The notebooks do not fabricate
public scores or add untrained rows to `submissions/experiment_log.csv`.

Notebook 11 is not automatically changed. Once these OOF results are reviewed,
add promising candidates to an explicitly versioned ensemble experiment so the
old submissions and their logged Kaggle scores keep referring to the same files.
Keep blend grids small and inspect error diversity rather than including every
available model. A weaker standalone model can still contribute, as TCN did.

Other plausible later ideas are fold-safe per-unit calibration and a small
residual sequence model, but cross-fitting must avoid training on in-sample
predictions or validation labels. A large Transformer/GNN, unrestricted neural
search, recent-year-only weighting, hard unit ordering, or strong overlap
smoothing is lower priority given current evidence. No new score is guaranteed.

## Verification actually performed

- Built the existing and added features on the real supplied data; checked
  finite values, dimensions and the nine source-reservoir mappings.
- Eight pure tests pass: within-case windows/runs, joint state/marginal ordering,
  purged inner timing, tunnel traversal, unit coverage and atomic checkpoints.
- All three notebooks pass notebook schema validation and Python syntax parsing,
  have no executed cells, and contain no imports from local `src` modules.
- Confirmed installed package versions: XGBoost 3.4.1, LightGBM 4.7.0,
  NumPy 2.5.3, pandas 3.0.6 and threadpoolctl 3.7.0.
- **No training, notebook execution, speed benchmark or Kaggle upload was run.**
  Model runtime behavior and accuracy must be verified when you run the notebooks.

Before the final course hand-in, select and trim the winning configuration,
record versions, retain report outputs and verify the specified 12-hour runtime
limit. The current notebooks are research experiments, not yet validated final
submission reproductions.
