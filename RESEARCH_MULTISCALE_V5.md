# Research audit and next experiment — 28 September 2026

## Bottom line

The best reported submission is now **physics_features_v4_blend.csv, 0.99005**.
XGBoost alone is 0.98996 and LightGBM 0.98968 (user screenshot). The original
physics blend was 0.98990. The v4 improvement is real on the reported public
subset, but small. The reported leader's 0.99562 would require reducing `1-AUC`
by about 56% relative to 0.99005. This is not a classification error rate and
does not imply that a 56% reduction in Brier loss would produce that score.
No experiment can honestly guarantee beating an unseen leaderboard result.

Notebook **18_multiscale_boosting.ipynb** tests two specific remaining gaps:
whole-week input representation and whether the inherited tree counts underfit.
It preserves the strong XGBoost/LightGBM family, includes the v4 control, and
separates feature changes from additional boosting. It does NOT train a huge
unbounded ensemble or deploy a diagnostic that already failed locally.

## What was inspected

- Raw task and dataset definitions, grading PDF, topology, mapping and submission
  schema; the shared pipeline and research/dispatch/refinement helpers.
- Notebooks 01–17, existing model/feature specifications, experiment log,
  saved forward predictions, feature gains, manifests and runtimes.
- Model outputs and scores are stronger evidence than a notebook's displayed
  execution counter. The TCN notebook contains an interrupted later run; do not
  attribute its currently displayed outputs automatically to its earlier CSV.
- `05_sequence_models.ipynb`, `06_ensemble_and_interpretation.ipynb`, and the
  current `final_submission` templates are not evidence of completed experiments.
  `Team_10` contains another task's examples, not hydro-model results.
- New scripts `scripts/audit_prediction_gaps.py` and
  `scripts/audit_feature_coverage.py` make the analyses reproducible without
  fitting the large case-hour boosters.

## Model inventory: what helped and what did not

Public scores below are user-reported, not a fresh authenticated Kaggle query.
Full recorded history is in `submissions/experiment_log.csv`, now including 16/17.

| Model / experiment | Public AUC | Assessment |
|---|---:|---|
| Historical climatology | — | Local mean AUC 0.54351; useful sanity baseline only. |
| Logistic / LightGBM, 20 features | 0.82402 / 0.82273 | The raw predictor family alone was not enough. |
| Engineered logistic / LightGBM | 0.97438 / 0.98061 | Feature representation produced the largest early gains. |
| Logistic + LightGBM blend | 0.98381 | Complementarity helped at this stage. |
| CatBoost per unit / pooled | 0.98549 / 0.98334 | Competitive; pooled version sampled 35%, not a fair full-data architecture comparison. |
| Original XGBoost per unit | 0.98824 | Strongest original single model. |
| EBM, sampled | 0.78948 | This configuration underfit badly; not evidence all GAMs are unsuitable. |
| Multi-output ExtraTrees | 0.94501 | Sharing outputs alone did not solve the task. |
| Multitask TCN | 0.92747 | This particular sequence model was weaker; more sequence capacity is not automatically better. |
| Greedy OOF / logistic stack | 0.98774 / 0.98716 | Stacking the old pool did not beat its strongest member publicly. |
| Temporal ensemble | 0.98825 | Small improvement over original XGB, not evidence smoothing always helps. |
| Physics XGB / LGBM | 0.98979 / 0.98957 | Domain features helped both model families. |
| Joint plant-state LightGBM | 0.98934 | Useful structural attempt; did not beat the physics singles. |
| Physics 50/50 blend | 0.98990 | Strong reference. |
| Dispatch experimental, notebook 16 | 0.98643 | Regressed; feature and capacity/stopping changes were confounded. |
| Dispatch guarded output | 0.98990 | Baseline fallback, NOT a new improved model. |
| Competition XGB / LGBM, notebook 17 | 0.98996 / 0.98968 | Small, controlled input improvement. |
| Competition 50/50 blend | **0.99005** | Current best reported result. |

Local scores across all notebooks are not directly comparable: validation years,
early-stopping treatment and fixed/refitted tree counts differ. Prefer matched
comparisons within a single experiment. Old counts themselves were selected
using historical validation; later replay does not make those years untouched.

## Feature coverage and measured effects

The actual v4 estimator has **280 columns per generator**: 175 original engineered,
75 global physics, 18 local physics, and 12 competition features. An inventory of
every feature for each of the nine plants and both audit/test periods is saved in
`artifacts/prediction_gap_audit/all_v4_feature_diagnostics.csv` (5,040 rows).
Saved per-unit/model gain entries are in `v4_all_feature_gains_2022.csv` (7,840 rows).
These are descriptive diagnostics for all inputs, not 280 causal ablation tests.

| Feature family | Already present | Evidence / remaining gap |
|---|---|---|
| Calendar | Hour, weekday, month cycles; horizon position | Useful context; cannot alone represent economic regime changes. |
| Price | Current price, weekly rank/mean/std, local lags/leads and moving means | Broad coverage, but no explicit vector of all seven days' price opportunities. |
| Water economics | Per-reservoir terminal values, margins and interactions | Dominant saved gains. Preserve these; do not misinterpret local water values as needing a second downstream subtraction. |
| Hydrology | Initial fills, all supplied reservoir/river inflows, weekly aggregates | No future observed volumes are needed or used. Some plant-source summaries are approximations. |
| Constraints | Flow/volume minima, initial slack, active indicators | Already covered. Static constraints alone cannot fix dispatch dynamics. |
| Physics | Within-case cumulative/remaining inflow, no-release bounds, source head, margin runs/profit windows | Strong improvement historically. No-release bounds are proxies, not actual reservoir states. |
| Better-hour competition | Better-price counts/premiums before/after each hour | Small positive result. Counts do not preserve whether competing hours are contiguous. |
| Dispatch proxies | Connected storage, estimated discharge, dynamic-program proxy, broader catchment | Tested in 16 with simultaneous capacity changes; cannot blame the entire feature family from that single comparison. |
| Inflow/profit timing additions | Arrival/profit fractions, future floors, remaining peaks | In controlled 17 development, adding this whole group hurt relative to competition-only for both backends. |

Notebook-17 matched 2021 results:

| Features | XGB | LGBM |
|---|---:|---:|
| Physics v2 | 0.984074 | 0.983760 |
| + competition | **0.984193** | **0.983922** |
| + competition + timing | 0.984088 | 0.983807 |

On the reused 2022 audit, the new blend scored **0.986236**, versus **0.986219**
for the original physics blend: only +0.000017. This is weak evidence of a large
generalization improvement, despite the subsequent modest public improvement.

The largest average normalized gains in the saved v4 audit models belong to
`local_margin_normalized` (~0.117), `local_margin` (~0.117), and
`price_margin__Vinjevatn` (~0.097). Correlated inputs split credit unevenly; gain
is not a causal measure and should not be used to delete all low-gain columns.

Input-only distribution checks expose a substantial **2022 economic shift**:
the median absolute standardized mean shift of the price group is ~5.42 relative
to pre-2022 rows; water-margin group ~2.62. Corresponding test-input shifts are
~1.09 and ~0.23. Hydrology shifts are much smaller on this aggregate diagnostic.
This helps explain why raw-price thresholds and a calibrator fitted to 2021 can
transfer poorly. It does not identify which hidden test subset Kaggle scores.
The normalized shape descriptors in 18 are a hypothesis for greater robustness.

## Error EDA and small tests actually run

All the following use saved forward v4 predictions. No large booster was trained.
The linear probes used only purged 2021 OOF rows for fitting, then evaluated 2022.

| Prediction / diagnostic | 2022 micro AUC |
|---|---:|
| v4 50/50 baseline | **0.986236** |
| Per-unit logistic calibration | 0.983662 |
| Linear stack with nearby-time and sibling-unit context | 0.983884 |
| Small case-level weekly-budget forest correction | 0.984570 |

The budget probe was a single 32-tree, depth-limited multi-output forest on about
2,550 weekly examples, not millions of case-hour rows. Its weekly ON-fraction
MAE was 0.17655, versus 0.05296 for the existing hourly model's weekly mean.
It is therefore NOT included in the new ensemble. Calibration/context probes
also failed, so we do not promote them merely because stacking sounds stronger.

For diagnosis only, shifting each case/unit's logit intercept to match its **true**
weekly ON fraction gave AUC 0.996530. This is a label-informed oracle, NOT a valid
model, submission, feasible target guarantee, or estimate of attainable Kaggle
score. It only shows that errors in whole-week operating levels can dominate;
the failed budget probe shows that learning those levels is not trivial. Oracle
code exists only in the offline diagnostic script, never in notebook 18 features,
selection, final predictions or the exported short notebook.

Hardest v4 audit units:

| Unit | AUC | Brier error | Error share within ±2h of a true transition |
|---|---:|---:|---:|
| Lio G1 | 0.961406 | 0.079327 | 24.9% |
| Haukeli G1 | 0.938574 | 0.067614 | 20.4% |
| Hogga G1 | 0.978454 | 0.057693 | 36.8% |
| Vinje G3 | 0.974652 | 0.051645 | 55.4% |
| Vinje G2 | 0.981405 | 0.050369 | 52.2% |

Lio/Haukeli need more than transition smoothing. Tokke G1 is already 0.994696;
Byrte 0.998975. Per-unit AUCs are not additive contributions to global micro AUC.
Day 1 of the horizon is worse (0.983846) than day 7 (0.988659). April (0.960793)
and November (0.967970) are weaker than February (0.993543) or August (0.996242).
These observations motivate case/horizon-aware evaluation; they are not used
for per-month label-dependent postprocessing. Full unit diagnostics are saved in
`unit_gaps_2022.csv`, and probe scores in the neighboring CSV files.

## External research and decisions

- [XGBoost's tuning guide](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
  distinguishes complexity and regularization from extra boosting, and warns
  about data copies/parallel experiment memory. We keep depth/regularization
  fixed, compare boosting prefixes, and fit only one model at a time. We do not
  apply per-unit class balancing blindly: the competition compares scores across
  units, so changes to their probability scales can affect global micro AUC.
- [LightGBM's tuning guide](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html)
  highlights leaf count, leaf sample count, and the learning-rate/tree-count
  tradeoff. Merely doubling leaves can overfit; we first test more iterations at
  the successful 31-leaf configuration. Thread count is explicit, not nested.
- [CatBoost tuning](https://catboost.ai/docs/en/concepts/parameter-tuning)
  supports depth/regularization tuning and special quantization for important
  numerical features. A physics-feature CatBoost is a sensible later diversity
  candidate: the old CatBoost did not receive the full newer feature set.
  It is not included in this first bounded experiment, to avoid adding a third
  expensive family before testing the stronger pair's representation/counts.
- [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html)
  requires independent fitting predictions. A shared monotone transform does
  not improve AUC ranking by itself; per-unit transforms can change global
  ranking but can also fail under shift. Our purged transfer probe did fail.
- [scikit-learn stacking](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
  warns about fitting a stack on in-sample predictions from already trained
  bases. We do not feed true sibling commitments or in-sample base predictions
  as shortcut features. A future nonlinear stack would require proper forward
  base predictions and additional cold-run training cost.
- [SINTEF generator documentation](https://docs.shop.sintef.energy/objects/generator/generator.html)
  describes head/discharge/efficiency and switching costs. Whole-case state and
  contiguous price opportunities are plausible determinants. This is a domain
  inference, not proof the new descriptors improve AUC. Current SHOP docs may
  postdate the dataset's version; no claim to replicate its optimizer is made.

Pretrained external models/data are not an appropriate shortcut under the course's
no-external-data and offline-reproduction requirements. Larger sequence models,
unbounded parameter searches, and leaderboard-specific weight hunting were not
chosen for this CPU/time-constrained next experiment.

## Notebook 18: exact experiment and reproduction

- New group: **54 features per plant** (30 common price-shape, 24 local economics/
  inflow descriptors). Fixed DCT, no learned transform; all windows stay inside
  each supplied 168-hour case. Volume remains initial-only; terminal water value
  follows the supplied definition. Do not use solver runtime, run ID as a feature,
  observed future storage, or commitment labels as predictors.
- Development: four backend/feature fits per unit (v4/path × XGB/LGBM), each to
  2x its inherited count. Retain prefixes at 1x, 1.5x, 2x. **56 unit fits**, not 168.
  No simultaneous change in depth, learning rate, regularization, or seeds.
- Select best variant/count multiplier per backend by 2021 micro AUC, then one
  global XGB weight from 0.25/0.5/0.75. Ties favor existing/smaller settings.
- Frozen 2022 audit vs the exact v4 50/50 control, at most 56 unit fits. This is
  reused historical validation, not an untouched test. Do not retune on audit
  errors and then describe that same audit as independent evidence.
- Final: **28 unit fits**, training all labeled data; writes the blend and both
  individual predictions. The candidate is experimental if the audit fails.
  Keep the existing 0.99005 submission as fallback. If original 1x v4 wins the
  controlled selection, the output can legitimately be the old pair.
- Local profile: default 12 CPU threads, one model at a time, BLAS one thread,
  float32 arrays; per-unit atomic checkpoints. Native cache identities include
  data, source, versions, parameters, split, tree counts, and thread count.
- Compatible v4 baseline caches can be read locally after strict provenance
  checks. Grading never requires these caches or the previous notebook.
- Exported short notebook: all code embedded, frozen small configuration,
  **four-thread CPU cold run**, no development/audit fits, no restored predictions.
  It starts from supplied raw data and mapping/sample files. Package versions
  must match the selection; source/data/version changes fail clearly.

The grading PDF permits fixed hyperparameters with documented provenance,
requires offline raw-data reproduction, and states max 12 hours on a normal PC
(approximately max four cores / 32 GB). This design is compatible in structure,
but **a four-core end-to-end runtime/RAM measurement remains the user's training
step**, not a completed guarantee. The old v4 final artifacts record ~263 s XGB
and ~343 s LGBM on the machine/profile that produced those runs; those timings
are not professor-PC benchmarks and cannot simply be promised for the new model.
No full-scale model in 18 was trained by the assistant.

After running: inspect `development_scores.csv`, `selection.json`,
`audit_diagnostics.csv`, per-unit importances/runtimes, then submit
`submissions/multiscale_boosting_v5.csv` if warranted. The last cell creates
`artifacts/multiscale_boosting_v5/Short_notebook_multiscale_v5.ipynb`.
Use that generated notebook, not the research notebook, for eventual reproduction.
Final Canvas report/two selected notebooks are a separate packaging task; the
existing report templates have not been silently treated as finished.

## Construction verification (not candidate training)

- Entire test directory: **22 passed**. New checks cover exact contiguous-run
  arithmetic, case-boundary isolation, constant-price paths, tree-prefix API
  behavior for both libraries on 64-row toy data, cache invalidation, constant
  targets, selection tie-breaking, and legacy cache provenance.
- Executed the actual exporter in a temporary directory; parsed its JSON/code,
  executed its embedded definitions, verified professor mode/cache disabling,
  and exercised its complete final/submission path with toy data and a fake
  estimator. No helper-module dependency or previous notebook remains.
- Executed notebook-18 setup/definitions/feature-building cells on the real raw
  data without running any training cells. Build took ~5.14 seconds; process
  peak RSS ~3.57 GB. This is NOT peak memory during training.
- Both notebook-17 competition audit caches matched all implemented provenance
  checks. Details in `artifacts/multiscale_boosting_v5/construction_checks.json`.
- The final boosted models, their Kaggle score, and the professor cold runtime
  remain deliberately unmeasured until the user runs the notebook.
