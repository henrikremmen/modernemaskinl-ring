"""Lightweight research helpers for notebook 20. Importing never trains or solves LPs.

Only stored out-of-time predictions and supplied case inputs enter features.
The correction is a bounded, constant logit shift per case/unit: no timing model.
"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

HORIZON = 168


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_probe_inputs(root):
    """Read only necessary arrays; refuse mismatched notebook-19 results or missing cache."""
    from src.sintef_challenger_runner import raw_digest
    root = Path(root)
    folder = root / 'artifacts/hydro_lp_v6'
    summaries, stored, paths = {}, {}, []
    for year, stage in [(2020, 'development'), (2021, 'development'), (2022, 'audit')]:
        summary = folder / f'summary_{stage}_{year}.json'
        archive = folder / f'predictions_{stage}_{year}.npz'
        if not summary.exists() or not archive.exists():
            raise FileNotFoundError(f'Finish and save notebook 19 first: {summary}, {archive}')
        summaries[year] = json.loads(summary.read_text())
        with np.load(archive, allow_pickle=False) as z:
            stored[year] = {k: z[k].copy() for k in ('case_ids', 'candidate_v5_lp', 'control_v5')}
        paths += [summary, archive]
    identity = summaries[2020]['identity']
    if any(s['identity'] != identity for s in summaries.values()):
        raise ValueError('Notebook-19 years have different identities; do not combine them.')
    if identity['raw'] != raw_digest(root):
        raise ValueError('Raw data no longer matches the notebook-19 results.')
    cache = folder / f"lp_cube_{identity['source'][:12]}_{identity['raw'][:12]}.npz"
    if not cache.exists():
        raise FileNotFoundError(f'Missing exact LP cache {cache}. No LP is recomputed by this probe.')
    with np.load(cache, allow_pickle=False) as z:
        cube, plants = z['cube'].copy(), list(z['plants'].astype(str))
    target_path = root / 'data/kernel/Unit_commitment_decisions.csv'
    header = pd.read_csv(target_path, nrows=0).columns
    columns = [c for c in header if re.fullmatch(r'result_committed_(.+)_t\d+', c)]
    units = list(dict.fromkeys(re.fullmatch(r'result_committed_(.+)_t\d+', c).group(1) for c in columns))
    expected_columns = [f'result_committed_{u}_t{h}' for u in units for h in range(HORIZON)]
    if columns != expected_columns or len(units) != 14:
        raise ValueError('Unexpected target layout.')
    targets = pd.read_csv(target_path, usecols=['Run No', 'starttime'] + columns,
                          dtype={c: np.int8 for c in columns})
    starts = pd.DatetimeIndex(pd.to_datetime(targets.starttime, utc=True))
    if not starts.is_monotonic_increasing or starts.has_duplicates:
        raise ValueError('Cases must be chronological and unique.')
    truth = targets[columns].to_numpy().reshape(len(targets), len(units), HORIZON)
    if not np.isin(truth, [0, 1]).all():
        raise ValueError('Non-binary targets.')
    n_test = len(pd.read_csv(root / 'prediction_mapping.csv', usecols=['Run No']))
    if cube.shape != (len(targets) + n_test, len(plants), 6, HORIZON) or not np.isfinite(cube).all():
        raise ValueError('Invalid LP cache shape or values.')
    topology = yaml.safe_load((root / 'data/extended/Tokke_Vinje_topology.yaml').read_text())
    price_frame = pd.read_csv(root / 'data/kernel/Historical_day_ahead_price_2015_2025.csv')
    price_series = price_frame.set_index(pd.to_datetime(price_frame.date, utc=True))['Day-ahead price (EUR/MWh)']
    bundles = {}
    for year, z in stored.items():
        ids = z.pop('case_ids')
        expected = np.flatnonzero(starts.year == year)
        if not np.array_equal(ids, expected):
            raise ValueError(f'{year}: wrong case IDs/order.')
        y = truth[ids].copy()
        for name, p in z.items():
            if p.shape != y.shape or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
                raise ValueError(f'{year}/{name}: invalid predictions.')
            key = 'candidate' if name == 'candidate_v5_lp' else 'control'
            if abs(roc_auc_score(y.ravel(), p.ravel()) - summaries[year][key]) > 1e-9:
                raise ValueError(f'{year}/{name}: predictions do not match saved score.')
        times = starts[ids].repeat(HORIZON) + pd.to_timedelta(np.tile(np.arange(HORIZON), len(ids)), unit='h')
        price = price_series.reindex(times).to_numpy().reshape(len(ids), HORIZON)
        if not np.isfinite(price).all():
            raise ValueError('Missing supplied hourly prices.')
        bundles[year] = dict(case_ids=ids, starts=starts[ids], truth=y, base=z['candidate_v5_lp'],
                             v5=z['control_v5'], lp=cube[ids].copy(), price=price)
    provenance = dict(notebook19_identity=identity,
                      input_sha256={str(p.relative_to(root)): file_digest(p) for p in paths + [cache]})
    return bundles, units, plants, topology, provenance


def purged_training_mask(train_starts, validation_starts):
    train, valid = pd.DatetimeIndex(train_starts), pd.DatetimeIndex(validation_starts)
    if not len(train) or not len(valid):
        raise ValueError('Empty split.')
    keep = np.asarray(train + pd.Timedelta(hours=HORIZON - 1) < valid.min())
    if not keep.any():
        raise ValueError('No training weeks remain after purge.')
    return keep


def run_statistics(margin, power, switching_cost):
    """Per-week duration and rough net value of positive-margin runs. No labels.

    price minus LP dual threshold is an opportunity-cost proxy, not SHOP profit.
    A run crossing a horizon edge is truncated; no cross-case state is assumed.
    """
    rows = []
    for m, p in zip(margin, power):
        active = m > 0
        edges = np.diff(np.r_[False, active, False].astype(int))
        a, b = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        lengths = b - a
        profits = np.array([np.sum(m[x:z] * p[x:z]) for x, z in zip(a, b)])
        paid = profits > switching_cost
        rows.append([len(a), lengths.max() if len(a) else 0,
                     lengths.mean() if len(a) else 0,
                     lengths[paid].sum() / HORIZON,
                     np.log1p(np.maximum(profits - switching_cost, 0).sum() / max(switching_cost, 1))])
    return np.asarray(rows, dtype=float)


def weekly_features(bundle, units, plants, topology):
    """Returns matched control/enhanced feature frames for each generator; never reads truth."""
    control, enhanced = {}, {}
    gens = topology['model']['generator']
    for i, u in enumerate(units):
        p, old = bundle['base'][:, i], bundle['v5'][:, i]
        qtiles = np.quantile(p, [.1, .5, .9], axis=1)
        base = pd.DataFrame(dict(lp_model_mean=p.mean(1), lp_model_std=p.std(1),
            lp_model_q10=qtiles[0], lp_model_q50=qtiles[1], lp_model_q90=qtiles[2],
            uncertainty=(p * (1 - p)).mean(1), soft_daily_variation=p.reshape(-1, 7, 24).mean(2).std(1),
            temporal_variation=np.abs(np.diff(p, axis=1)).mean(1),
            old_v5_mean=old.mean(1), old_v5_disagreement=(old-p).mean(1)))
        plant = u.rsplit('_G', 1)[0]
        j = plants.index(plant)
        q, threshold, rc, src, dst, conversion = (bundle['lp'][:, j, k] for k in range(6))
        members = [g for name, g in gens.items() if name.rsplit('_G', 1)[0] == plant]
        qmax = sum(max(max(c['x']) for c in g['turb_eff_curves']) for g in members)
        capacity = sum(g['p_max'] for g in members)
        # Approximate plant MW from LP discharge, allocated to units by rated capacity.
        plant_mw = np.maximum(q, 0) * qmax * conversion
        share_mw = plant_mw * float(gens[u]['p_max']) / capacity
        pmin = float(gens[u]['p_min'])
        pmax = float(gens[u]['p_max'])
        margin = bundle['price'] - threshold
        scale = np.abs(bundle['price']).mean(1) + 1
        run = run_statistics(margin, np.minimum(share_mw, pmax),
            float(gens[u].get('startcost_const', 0) + gens[u].get('stopcost_const', 0)))
        extra = pd.DataFrame(dict(lp_week_discharge=q.mean(1), lp_daily_variation=q.reshape(-1,7,24).mean(2).std(1),
            lp_active_fraction=(q > .01).mean(1), lp_at_capacity_fraction=(q > .99).mean(1),
            lp_week_vs_predicted=q.mean(1)-p.mean(1),
            allocated_mw_below_min_fraction=((share_mw > 1e-6) & (share_mw < pmin)).mean(1),
            allocated_mw_over_max_fraction=(share_mw > pmax).mean(1),
            allocated_mw_over_pmin_mean=(share_mw / max(pmin, 1e-6)).mean(1),
            plant_min_units_proxy=(plant_mw / max(g['p_max'] for g in members)).mean(1),
            dual_positive_fraction=(margin > 0).mean(1),
            dual_positive_mean=np.maximum(margin, 0).mean(1)/scale,
            reduced_margin_mean=rc.mean(1)/scale,
            source_relative_range=np.ptp(src, axis=1)/(np.abs(src[:,0])+1),
            downstream_relative_range=np.ptp(dst, axis=1)/(np.abs(dst[:,0])+1),
            positive_run_count=run[:,0], longest_positive_run=run[:,1], mean_positive_run=run[:,2],
            estimated_paid_run_fraction=run[:,3], log_estimated_net_run_value=run[:,4]))
        control[u], enhanced[u] = base, pd.concat([base, extra], axis=1)
        if not np.isfinite(enhanced[u].to_numpy()).all():
            raise ValueError(f'Nonfinite features for {u}.')
    return {'control': control, 'enhanced': enhanced}


def level_shift(probabilities, desired_mean, max_logit_shift=1.5):
    """Bounded constant shift preserving within-week ordering; returns probabilities and shifts."""
    p = np.asarray(probabilities, dtype=float)
    target = np.asarray(desired_mean, dtype=float)
    if p.ndim != 2 or target.shape != p.shape[:1] or max_logit_shift <= 0:
        raise ValueError('Bad correction dimensions/bound.')
    if not np.isfinite(p).all() or not np.isfinite(target).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid correction inputs.')
    z = logit(np.clip(p, 1e-7, 1-1e-7))
    low, high = np.full(len(p), -max_logit_shift), np.full(len(p), max_logit_shift)
    for _ in range(35):
        mid = (low+high)/2
        below = expit(z+mid[:,None]).mean(1) < target
        low, high = np.where(below, mid, low), np.where(below, high, mid)
    shift = (low+high)/2
    result = expit(z+shift[:,None])
    unchanged = np.abs(target - p.mean(1)) < 1e-10
    result[unchanged], shift[unchanged] = p[unchanged], 0
    return result.astype(np.float32), shift


def run_probe(train, valid, train_features, valid_features, units,
              alpha=100., shrinkage=.25, max_level_change=.10, max_logit_shift=1.5, threads=4):
    """28 tiny ridge fits. All preprocessing fits ONLY on purged training weeks.

    No evaluation labels are read by this function. Hyperparameters are frozen inputs.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from threadpoolctl import threadpool_limits
    if alpha <= 0 or not 0 <= shrinkage <= 1 or not 0 < max_level_change <= 1:
        raise ValueError('Invalid probe configuration.')
    keep = purged_training_mask(train['starts'], valid['starts'])
    outputs, records, coefficients = {}, [], []
    for arm in ('control', 'enhanced'):
        corrected = np.empty_like(valid['base'])
        for i, u in enumerate(units):
            x, v = train_features[arm][u], valid_features[arm][u]
            if list(x.columns) != list(v.columns):
                raise ValueError('Feature columns changed across years.')
            residual = train['truth'][:, i].mean(1)-train['base'][:, i].mean(1)
            with threadpool_limits(limits=max(1, min(int(threads), 4))):
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='cholesky'))
                model.fit(x.loc[keep], residual[keep])
                delta = np.clip(shrinkage*model.predict(v), -max_level_change, max_level_change)
            desired = np.clip(valid['base'][:, i].mean(1)+delta, 1e-6, 1-1e-6)
            corrected[:, i], shifts = level_shift(valid['base'][:, i], desired, max_logit_shift)
            records.append(dict(arm=arm, unit=u, train_weeks=int(keep.sum()), purged=int((~keep).sum()),
                                mean_abs_shift=float(np.abs(shifts).mean()),
                                shift_at_bound_fraction=float((np.abs(shifts) > max_logit_shift-1e-5).mean())))
            coefficients.extend(dict(arm=arm,unit=u,feature=f,coefficient=float(c))
                for f,c in zip(x.columns,model[-1].coef_))
        outputs[arm] = corrected
    return outputs, pd.DataFrame(records), pd.DataFrame(coefficients)


def score_predictions(truth, predictions, units, starts):
    rows = []
    starts = pd.DatetimeIndex(starts)
    for name, p in predictions.items():
        if p.shape != truth.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
            raise ValueError('Invalid evaluation prediction.')
        def add(group, label, y, q):
            rows.append(dict(model=name,group=group,label=str(label),
                micro_auc=float(roc_auc_score(y.ravel(),q.ravel())) if np.unique(y).size==2 else np.nan,
                brier=float(np.mean((q-y)**2)),weekly_level_mae=float(np.abs(q.mean(-1)-y.mean(-1)).mean())))
        add('all','all',truth,p)
        for i,u in enumerate(units): add('unit',u,truth[:,i],p[:,i])
        for month in np.unique(starts.month):
            mask=starts.month==month
            add('month',month,truth[mask],p[mask])
    return pd.DataFrame(rows)


def error_diagnostics(bundle, units):
    """Labels are used for diagnostics ONLY, never for feature construction."""
    y,p=bundle['truth'],bundle['base']
    rows=[]
    for i,u in enumerate(units):
        yi,pi=y[:,i],p[:,i]
        transition=np.zeros_like(yi,dtype=bool)
        change=yi[:,1:] != yi[:,:-1]
        transition[:,:-1] |= change
        transition[:,1:] |= change
        mixed=(yi.min(1)!=yi.max(1))
        for label,mask in [('near_true_transition',transition),('away_from_transition',~transition),
                           ('mixed_weeks',np.broadcast_to(mixed[:,None],yi.shape)),
                           ('constant_weeks',np.broadcast_to(~mixed[:,None],yi.shape))]:
            rows.append(dict(unit=u,group=label,hours=int(mask.sum()),
                brier=float(np.mean((pi[mask]-yi[mask])**2)) if mask.any() else np.nan))
    return pd.DataFrame(rows)
