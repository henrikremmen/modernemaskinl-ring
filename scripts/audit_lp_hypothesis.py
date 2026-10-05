"""Evidence behind notebook 19, from saved notebook-18 predictions. No base-model training.

Writes artifacts/hydro_lp_v6/hypothesis_audit/*.csv:
  level_oracle.csv   AUC if each case/unit's weekly ON-fraction were known (label-informed ceiling;
                     constant case/units become free), plus v5 AUC within mixed ON/OFF weeks
  rule_vs_lp.csv     untrained AUC per unit/year: price - water value versus LP reduced-cost margin
  lio_min_flow.csv   Lio ON-rate below its water value, split by r_Vest_Vassdraget minimum-flow deficit
  level_compare.csv  2022 weekly-level MAE: v5 alone, LP alone, and a linear level model fitted on
                     2021 using v5 level only versus v5 + LP level (incremental, out of sample)
  stack_probe.csv    small per-unit LightGBM stacker fitted on 2021, scored 2022, with/without LP inputs
"""
import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sintef_pipeline import build_project_data, HORIZON
from src.sintef_challenger_runner import purged_year
from src.sintef_hydro_lp import solve_all_cases, PLANT_SOURCE

V5 = ROOT / 'artifacts/multiscale_boosting_v5'
OUT = ROOT / 'artifacts/hydro_lp_v6/hypothesis_audit'


def v5_development(fam):
    folder = glob.glob(str(V5 / f'development_2021_{fam}_path/*/'))[0]
    return np.stack([np.load(f'{folder}/unit_{i:02d}.npz')['predictions'][2] for i in range(14)], axis=1)  # 2x


def level_oracle_auc(y, p):
    z = logit(np.clip(p, 1e-5, 1 - 1e-5))
    target, lo = y.mean(axis=2), np.full(y.shape[:2], -20.)
    hi = -lo
    for _ in range(40):
        mid = (lo + hi) / 2
        low = expit(z + mid[:, :, None]).mean(axis=2) < target
        lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
    return roc_auc_score(y.ravel(), expit(z + ((lo + hi) / 2)[:, :, None]).ravel())


def within_week_auc(y, p):
    """Timing only: AUC inside each case/unit with both classes, weighted by positive x negative pairs."""
    total = weight = 0.
    for c in range(len(y)):
        for i in range(y.shape[1]):
            pos = y[c, i].sum()
            if 0 < pos < y.shape[2]:
                w = pos * (y.shape[2] - pos)
                total += w * roc_auc_score(y[c, i], p[c, i])
                weight += w
    return total / weight


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_project_data(ROOT)
    n, N = data.n_all_cases, data.n_train_cases
    starts = pd.concat([data.train_cases, data.test_cases]).starttime
    cube, plants, seconds = solve_all_cases(ROOT, starts, threads=4)
    print(f'LP solved in {seconds:.0f}s')
    price = data.X_all['price'].to_numpy().reshape(n, HORIZON)
    wv = {p: data.X_all[f'water_value_end__{PLANT_SOURCE[p]}'].to_numpy().reshape(n, HORIZON) for p in plants}
    y_all = data.target_cube.astype(float)
    _, v21 = purged_year(data, 2021)
    _, v22 = purged_year(data, 2022)
    p21 = .5 * (v5_development('xgb') + v5_development('lgb'))
    p22 = np.load(V5 / 'audit_predictions.npz')['candidate']

    rows = [dict(year=yr, v5_auc=roc_auc_score(y_all[ids].ravel(), p.ravel()), oracle_level_auc=level_oracle_auc(y_all[ids], p),
                 constant_case_unit_share=(y_all[ids].std(axis=2) == 0).mean(), within_week_auc=within_week_auc(y_all[ids], p))
            for yr, ids, p in ((2021, v21, p21), (2022, v22, p22))]
    pd.DataFrame(rows).to_csv(OUT / 'level_oracle.csv', index=False)

    years = pd.DatetimeIndex(data.train_cases.starttime).year.to_numpy()
    rows = []
    for i, unit in enumerate(data.units):
        p, j = unit.rsplit('_G', 1)[0], plants.index(unit.rsplit('_G', 1)[0])
        rule = (price - wv[p])[:N]
        lp_score = cube[:N, j, 2] + 1e-3 * (price[:N] - cube[:N, j, 1])
        for yr in range(2015, 2023):
            s = years == yr
            y = y_all[s, i]
            rows.append(dict(unit=unit, year=yr, rule_auc=roc_auc_score(y.ravel(), rule[s].ravel()),
                             lp_auc=roc_auc_score(y.ravel(), lp_score[s].ravel())))
    pd.DataFrame(rows).to_csv(OUT / 'rule_vs_lp.csv', index=False)

    deficit = (data.X_all['min_flow__r_Vest_Vassdraget'] - data.X_all['inflow__r_Vest_Vassdraget']).to_numpy().reshape(n, HORIZON)[:N]
    i = data.units.index('Lio_G1')
    below = (price - wv['Lio'])[:N] <= 0
    rows = []
    for yr in range(2015, 2023):
        s = (years == yr)[:, None] & below
        y, d = y_all[:, i][s], deficit[s]
        rows.append(dict(year=yr, hours_below_value=int(s.sum()), on_rate=y.mean(),
                         on_rate_deficit=y[d > 0].mean(), on_rate_no_deficit=y[d <= 0].mean()))
    pd.DataFrame(rows).to_csv(OUT / 'lio_min_flow.csv', index=False)

    # Do not correlate (truth - v5) with (LP - v5): both share the -v5 term. Compare errors directly,
    # then test whether the LP level adds information to the v5 level out of sample (2021 -> 2022).
    from sklearn.linear_model import LinearRegression
    y22 = y_all[v22]
    rows = []
    for i, unit in enumerate(data.units):
        j = plants.index(unit.rsplit('_G', 1)[0])
        truth = {2021: y_all[v21, i].mean(1), 2022: y22[:, i].mean(1)}
        v5_level = {2021: p21[:, i].mean(1), 2022: p22[:, i].mean(1)}
        lp_level = {2021: cube[v21, j, 0].mean(1), 2022: cube[v22, j, 0].mean(1)}
        only = LinearRegression().fit(v5_level[2021][:, None], truth[2021])
        both = LinearRegression().fit(np.c_[v5_level[2021], lp_level[2021]], truth[2021])
        mae = lambda q: np.abs(np.clip(q, 0, 1) - truth[2022]).mean()
        rows.append(dict(unit=unit, mae_v5=mae(v5_level[2022]), mae_lp_alone=mae(lp_level[2022]),
                         mae_linear_v5_only=mae(only.predict(v5_level[2022][:, None])),
                         mae_linear_v5_plus_lp=mae(both.predict(np.c_[v5_level[2022], lp_level[2022]]))))
    pd.DataFrame(rows).to_csv(OUT / 'level_compare.csv', index=False)

    # Probe: identical small stacker with and without LP inputs. Fit 2021 (purged), score 2022.
    keep = pd.DatetimeIndex(starts.iloc[v21]) + pd.Timedelta(hours=HORIZON - 1) < pd.Timestamp(starts.iloc[v22[0]])
    rank = data.X_all['price_week_percentile'].to_numpy().reshape(n, HORIZON)

    def features(ids, p, i, use_lp):
        pl = data.units[i].rsplit('_G', 1)[0]
        j = plants.index(pl)
        cols = [logit(np.clip(p[:, i], 1e-5, 1 - 1e-5)), np.broadcast_to(np.arange(HORIZON), (len(ids), HORIZON)),
                rank[ids], price[ids] - wv[pl][ids]]
        if use_lp:
            c = cube[ids, j]
            cols += [c[:, 2], price[ids] - c[:, 1], c[:, 0], np.broadcast_to(c[:, 0].mean(1, keepdims=True), c[:, 0].shape),
                     c[:, 3], c[:, 1] - wv[pl][ids]]
        return np.column_stack([np.asarray(v, np.float32).ravel() for v in cols])
    params = dict(objective='binary', learning_rate=.05, num_leaves=15, min_data_in_leaf=200, feature_fraction=.9,
                  bagging_fraction=.8, bagging_freq=1, lambda_l2=5., verbosity=-1, num_threads=4, seed=7)
    predictions = {'v5_saved': p22}
    for use_lp in (False, True):
        q = np.empty_like(p22)
        for i in range(14):
            model = lgb.train(params, lgb.Dataset(features(v21[keep], p21[keep], i, use_lp), y_all[v21[keep], i].ravel()), 300)
            q[:, i] = model.predict(features(v22, p22, i, use_lp)).reshape(len(v22), HORIZON)
        predictions['stack_with_lp' if use_lp else 'stack_control'] = q
    rows = [dict(model=k, micro_auc=roc_auc_score(y22.ravel(), v.ravel()),
                 **{u: roc_auc_score(y22[:, i].ravel(), v[:, i].ravel()) for i, u in enumerate(data.units)})
            for k, v in predictions.items()]
    pd.DataFrame(rows).to_csv(OUT / 'stack_probe.csv', index=False)
    for name in ('level_oracle', 'lio_min_flow', 'level_compare', 'stack_probe'):
        print(name, pd.read_csv(OUT / f'{name}.csv').round(4).to_string(index=False), sep='\n')
    print(pd.read_csv(OUT / 'rule_vs_lp.csv').groupby('unit')[['rule_auc', 'lp_auc']].mean().round(4))


if __name__ == '__main__':
    main()
