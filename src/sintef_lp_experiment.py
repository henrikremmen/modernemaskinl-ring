"""Controlled experiment: exact v5 model versus v5 + LP-relaxation features. Importing never trains.

Only the feature set differs between control and candidate: same estimators, parameters, seeds,
2x inherited tree counts and 50/50 XGBoost/LightGBM blend as multiscale_boosting_v5.csv.
Selection uses the development years; the audit year is looked at once, afterwards.
"""
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from src.sintef_pipeline import HORIZON, rows_for_cases, make_submission
from src.sintef_challenger_runner import raw_digest, save_json, purged_year, score_cube, fit_member
from src.sintef_path_runner import staged_predict, checked_cube, scaled_rounds, factor_key

VARIANTS = ('v5', 'v5_lp')
FAMILIES = ('xgb', 'lgb')
FACTORS = (1., 1.5, 2.)       # prefixes of one fit; 1x/1.5x are diagnostics only
DECISION_FACTOR = 2.          # v5 used 2x the inherited per-unit counts
XGB_WEIGHT = .5               # v5 blend weight; not re-tuned here
MAX_DEV_YEAR_LOSS = .0005     # pre-registered: no development year may lose more than this


def append_csv(path, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode='a', header=not Path(path).exists(), index=False)


class LPRunner:
    def __init__(self, data, extra, local, competition, paths, lp, specs, rounds, threads,
                 source_digest, directory, use_checkpoints=True):
        self.data, self.local, self.competition, self.paths, self.lp = data, local, competition, paths, lp
        self.specs, self.rounds = specs, rounds
        self.threads = max(1, min(int(threads), os.cpu_count() or 1))
        self.use_checkpoints = bool(use_checkpoints)
        self.source_digest = source_digest
        self.base_names = list(data.feature_sets['full_engineered']) + list(extra.columns)
        self.base = np.column_stack([data.X_all[data.feature_sets['full_engineered']].to_numpy(np.float32),
                                     extra.to_numpy(np.float32)])
        self.raw = raw_digest(data.root)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def frames(self, plant, variant):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown variant {variant}')
        blocks = [self.local[plant], self.competition[plant], self.paths[plant]]
        return blocks + [self.lp[plant]] if variant == 'v5_lp' else blocks

    def feature_names(self, plant, variant):
        names = self.base_names + [c for f in self.frames(plant, variant) for c in f.columns]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate model features')
        return names

    def identity(self):
        return dict(source=self.source_digest, raw=self.raw, specs=self.specs, rounds=self.rounds,
                    packages={p: importlib.metadata.version(p) for p in
                              ('numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'lightgbm')})

    def run(self, family, variant, train, predict, stage, factors=FACTORS):
        """One fit per unit, several tree-count prefixes. Completed units resume from disk."""
        train, predict = np.asarray(train, dtype=int), np.asarray(predict, dtype=int)
        if not len(train) or not len(predict) or len(np.intersect1d(train, predict)):
            raise ValueError('Empty or overlapping training/prediction cases')
        if train.max() >= self.data.n_train_cases or predict.max() >= self.data.n_all_cases or min(train.min(), predict.min()) < 0:
            raise ValueError('Case index outside dataset')
        factors = tuple(sorted({float(f) for f in factors}))
        names_hash = hashlib.sha256(json.dumps({p: self.feature_names(p, variant) for p in self.local}).encode()).hexdigest()
        manifest = dict(self.identity(), family=family, variant=variant, stage=stage, threads=self.threads,
                        features=names_hash, train=list(map(int, train)), predict=list(map(int, predict)),
                        factors=list(factors))
        signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:24]
        folder = self.directory / f'{stage}_{family}_{variant}' / signature
        folder.mkdir(parents=True, exist_ok=True)
        save_json(folder / 'manifest.json', manifest)
        cubes = {factor_key(f): np.empty((len(predict), len(self.data.units), HORIZON), np.float32) for f in factors}
        train_rows, pred_rows = rows_for_cases(train), rows_for_cases(predict)
        for i, unit in enumerate(self.data.units):
            started = time.perf_counter()
            counts = [scaled_rounds(self.rounds[family][unit], f) for f in factors]
            checkpoint = folder / f'unit_{i:02d}.npz'
            plant = unit.rsplit('_G', 1)[0]
            if self.use_checkpoints and checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    if saved['rounds'].tolist() != counts:
                        raise ValueError(f'Mismatched checkpoint: {checkpoint}')
                    predictions = saved['predictions'].copy()
                status = 'restored'
            else:
                names = self.feature_names(plant, variant)
                X = np.ascontiguousarray(np.column_stack(
                    [self.base] + [f.to_numpy(np.float32) for f in self.frames(plant, variant)]), dtype=np.float32)
                y = self.data.target_cube[train, i].ravel()
                print(f'{stage} | {family}/{variant} | {unit} | {max(counts)} trees, {X.shape[1]} features', flush=True)
                if len(np.unique(y)) == 1:
                    predictions = np.full((len(factors), len(predict), HORIZON), float(y[0]), np.float32)
                    gain = np.zeros(len(names))
                else:
                    with threadpool_limits(limits=self.threads), threadpool_limits(limits=1, user_api='blas'):
                        model, _ = fit_member(self.specs[family], X[train_rows], y, max(counts), self.threads)
                        pred_X = X[pred_rows]
                        predictions = np.stack([staged_predict(model, self.specs[family]['backend'], pred_X, c, self.threads)
                                                .reshape(len(predict), HORIZON) for c in counts]).astype(np.float32)
                        gain = model.feature_importances_ if self.specs[family]['backend'] == 'xgboost' \
                            else model.feature_importance(importance_type='gain')
                    del model, pred_X
                pd.DataFrame(dict(feature=names, gain=gain)).sort_values('gain', ascending=False).to_csv(
                    folder / f'importance_{i:02d}.csv', index=False)
                temporary = checkpoint.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, predictions=predictions, rounds=np.asarray(counts))
                os.replace(temporary, checkpoint)
                del X, y
                gc.collect()
                status = 'trained'
            if predictions.shape != (len(factors), len(predict), HORIZON) or not np.isfinite(predictions).all() \
                    or predictions.min() < 0 or predictions.max() > 1:
                raise ValueError(f'Invalid checkpoint: {checkpoint}')
            for k, f in enumerate(factors):
                cubes[factor_key(f)][:, i] = predictions[k]
            seconds = time.perf_counter() - started
            append_csv(self.directory / 'runtime_log.csv', [dict(
                stage=stage, family=family, variant=variant, unit=unit, status=status, seconds=round(seconds, 2),
                trees=max(counts), threads=self.threads, train_cases=len(train), signature=signature)])
            print(f'  {status}: {seconds:.1f}s ({i + 1}/{len(self.data.units)})', flush=True)
        return cubes


def group_diagnostics(data, cases, predictions, year):
    """Per unit/month/horizon/UTC-hour AUC and Brier, plus weekly-level MAE. Descriptive only."""
    truth = data.target_cube[cases].astype(np.float32)
    starts = pd.DatetimeIndex(data.train_cases.starttime.iloc[cases])
    hour_of_day = (starts.hour.to_numpy()[:, None] + np.arange(HORIZON)[None, :]) % 24
    blocks = [(0, 6), (6, 24), (24, 72), (72, 120), (120, 168)]
    rows = []

    def add(name, group, label, y, p, level=np.nan):
        auc = float(roc_auc_score(y.ravel(), p.ravel())) if 0 < y.mean() < 1 else np.nan
        rows.append(dict(year=year, model=name, group=group, label=label, auc=auc,
                         brier=float(np.mean((p - y) ** 2)), on_rate=float(y.mean()),
                         predicted_rate=float(p.mean()), weekly_level_mae=level))
    for name, p in predictions.items():
        add(name, 'all', 'all', truth, p, float(np.abs(p.mean(2) - truth.mean(2)).mean()))
        for i, unit in enumerate(data.units):
            add(name, 'unit', unit, truth[:, i], p[:, i], float(np.abs(p[:, i].mean(1) - truth[:, i].mean(1)).mean()))
        for month in range(1, 13):
            s = starts.month == month
            if s.any():
                add(name, 'start_month', month, truth[s], p[s])
        for a, b in blocks:
            add(name, 'horizon_hours', f'{a}-{b - 1}', truth[:, :, a:b], p[:, :, a:b])
        for h in range(0, 24, 3):
            m = np.broadcast_to(((hour_of_day >= h) & (hour_of_day < h + 3))[:, None, :], truth.shape)
            add(name, 'utc_hour', f'{h:02d}-{h + 2:02d}', truth[m], p[m])
    return pd.DataFrame(rows)


def block_bootstrap_delta(truth, candidate, control, block=7, reps=200, seed=0):
    """Paired bootstrap of AUC(candidate) - AUC(control), resampling blocks of consecutive cases
    (adjacent daily cases share 144 hours, so single cases are not independent)."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    starts = np.arange(0, n, block)
    deltas = []
    for _ in range(reps):
        pick = np.concatenate([np.arange(s, min(s + block, n)) for s in rng.choice(starts, len(starts))])
        y = truth[pick].ravel()
        deltas.append(roc_auc_score(y, candidate[pick].ravel()) - roc_auc_score(y, control[pick].ravel()))
    return np.percentile(deltas, [5, 50, 95]), float(np.mean(np.asarray(deltas) > 0))


def evaluate_year(runner, year, stage):
    """Four fits x 14 units for one purged forward year; returns and saves all comparisons."""
    train, valid = purged_year(runner.data, year)
    truth = runner.data.target_cube[valid]
    started = time.perf_counter()
    single, rows = {}, []
    for family in FAMILIES:
        for variant in VARIANTS:
            t0 = time.perf_counter()
            snapshots = runner.run(family, variant, train, valid, f'{stage}_{year}')
            seconds = time.perf_counter() - t0
            for factor, p in snapshots.items():
                rows.append(dict(stage=stage, year=year, model=f'{family}_{variant}', factor=float(factor),
                                 micro_auc=score_cube(truth, p), seconds=round(seconds, 1)))
            single[(family, variant)] = snapshots[factor_key(DECISION_FACTOR)]
    blends = {v: XGB_WEIGHT * single[('xgb', v)] + (1 - XGB_WEIGHT) * single[('lgb', v)] for v in VARIANTS}
    for v, p in blends.items():
        rows.append(dict(stage=stage, year=year, model=f'blend_{v}', factor=DECISION_FACTOR,
                         micro_auc=score_cube(truth, p), seconds=round(time.perf_counter() - started, 1)))
    table = pd.DataFrame(rows)
    append_csv(runner.directory / 'results_log.csv', table.to_dict('records'))
    predictions = {'control_v5': blends['v5'], 'candidate_v5_lp': blends['v5_lp']}
    predictions.update({f'{f}_{v}': single[(f, v)] for f in FAMILIES for v in VARIANTS})
    diagnostics = group_diagnostics(runner.data, valid, predictions, year)
    diagnostics.to_csv(runner.directory / f'diagnostics_{stage}_{year}.csv', index=False)
    interval, share = block_bootstrap_delta(truth, blends['v5_lp'], blends['v5'])
    summary = dict(stage=stage, year=year, control=score_cube(truth, blends['v5']),
                   candidate=score_cube(truth, blends['v5_lp']),
                   delta_xgb=score_cube(truth, single[('xgb', 'v5_lp')]) - score_cube(truth, single[('xgb', 'v5')]),
                   delta_lgb=score_cube(truth, single[('lgb', 'v5_lp')]) - score_cube(truth, single[('lgb', 'v5')]),
                   bootstrap_delta_p05_p50_p95=[float(v) for v in interval], bootstrap_share_positive=share,
                   identity=runner.identity())
    summary['delta'] = summary['candidate'] - summary['control']
    save_json(runner.directory / f'summary_{stage}_{year}.json', summary)
    np.savez_compressed(runner.directory / f'predictions_{stage}_{year}.npz', case_ids=valid,
                        **{k: v.astype(np.float32) for k, v in predictions.items()})
    print(f'{year}: control v5 {summary["control"]:.6f} | candidate v5+LP {summary["candidate"]:.6f} | '
          f'delta {summary["delta"]:+.6f} (block bootstrap 90%: {interval[0]:+.6f} .. {interval[2]:+.6f})', flush=True)
    return summary, table, diagnostics


def select_candidate(runner, years):
    """Pre-registered rule, development years only: mean delta > 0 and no year worse than the tolerance."""
    summaries = []
    for year in years:
        path = runner.directory / f'summary_development_{year}.json'
        if not path.exists():
            raise FileNotFoundError(f'Run development year {year} first')
        s = json.loads(path.read_text())
        if s['identity'] != runner.identity():
            raise ValueError(f'Development {year} used another configuration; rerun it')
        summaries.append(s)
    deltas = [s['delta'] for s in summaries]
    accepted = bool(np.mean(deltas) > 0 and min(deltas) > -MAX_DEV_YEAR_LOSS)
    selection = dict(development_years=list(years), deltas=deltas, mean_delta=float(np.mean(deltas)),
                     accepted=accepted, variant='v5_lp' if accepted else 'v5', factor=DECISION_FACTOR,
                     xgb_weight=XGB_WEIGHT, rule=f'mean delta > 0 and every year > -{MAX_DEV_YEAR_LOSS}',
                     identity=runner.identity())
    save_json(runner.directory / 'selection.json', selection)
    return selection


def load_checked(runner, name):
    """A saved selection/summary only counts if this exact data/code/parameter setup produced it."""
    path = runner.directory / name
    if not path.exists():
        raise FileNotFoundError(f'{name} is missing; run the validation cells first')
    saved = json.loads(path.read_text())
    if saved.get('identity') != json.loads(json.dumps(runner.identity())):
        raise ValueError(f'{name} was produced by another data/code/parameter setup; rerun validation')
    return saved


def verdict(selection, audit):
    """Four outcomes; a positive point estimate whose 90% interval reaches zero is only 'uncertain'."""
    lo, _, hi = audit['bootstrap_delta_p05_p50_p95']
    if not selection['accepted']:
        return 'NOT SUPPORTED', 'Development rule rejected the LP features. Keep v5 selected; the CSV is experimental.'
    if audit['delta'] <= 0:
        return 'NOT CONFIRMED', 'Chosen on development years but not better on the later check. Keep v5 selected.'
    if lo <= 0:
        return 'POSITIVE BUT UNCERTAIN', (f'Point estimate {audit["delta"]:+.6f}, but the 90% interval {lo:+.6f} .. {hi:+.6f} '
                                          'includes no gain or a loss. An upload is a test, not evidence. Keep v5 selected.')
    return 'SUPPORTED', ('Accepted on development years and positive on the later check with the 90% interval above zero. '
                         'Upload hydro_lp_v6.csv as a new attempt; keep v5 as the fallback until the private score is known.')


def verify_submission(path, sample_path):
    submission, sample = pd.read_csv(path), pd.read_csv(sample_path)
    if list(submission.columns) != list(sample.columns) or submission.shape != sample.shape:
        raise AssertionError('Submission columns/shape differ from sample_submission.csv')
    if not submission['Run No'].equals(sample['Run No']):
        raise AssertionError('Run No order differs from sample_submission.csv')
    values = submission.iloc[:, 1:].to_numpy()
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise AssertionError('Invalid probabilities')
    return submission.shape


def final_lp(runner, variant='v5_lp', factor=DECISION_FACTOR, xgb_weight=XGB_WEIGHT, filename='hydro_lp_v6.csv'):
    """Fit the chosen pair on all labelled cases and write single-model and blend CSVs."""
    train = np.arange(runner.data.n_train_cases)
    test = np.arange(runner.data.n_train_cases, runner.data.n_all_cases)
    blend = np.zeros((len(test), len(runner.data.units), HORIZON), np.float32)
    paths = []
    for family in FAMILIES:
        p = runner.run(family, variant, train, test, 'final', factors=(factor,))[factor_key(factor)]
        blend += (xgb_weight if family == 'xgb' else 1 - xgb_weight) * p
        path, _ = make_submission(runner.data, p, f'{family}_{filename}')
        paths.append(path)
    path, submission = make_submission(runner.data, checked_cube(blend, len(test), len(runner.data.units)), filename)
    for p in paths + [path]:
        print('Verified', p, verify_submission(p, runner.data.root / 'sample_submission.csv'), flush=True)
    return path, submission
