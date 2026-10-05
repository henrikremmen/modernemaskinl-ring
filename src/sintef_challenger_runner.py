"""Sequential, resumable challenger. Importing this module never trains models."""
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


def raw_digest(root):
    paths = sorted((root / 'data').rglob('*.csv')) + sorted((root / 'data').rglob('*.yaml'))
    paths += [root / 'prediction_mapping.csv', root / 'sample_submission.csv']
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def purged_year(data, year):
    starts = pd.DatetimeIndex(data.train_cases['starttime'])
    begin = pd.Timestamp(f'{year}-01-01', tz='UTC')
    end = pd.Timestamp(f'{year+1}-01-01', tz='UTC')
    train = np.flatnonzero(starts + pd.Timedelta(hours=HORIZON-1) < begin)
    valid = np.flatnonzero((starts >= begin) & (starts < end))
    if not len(train) or not len(valid):
        raise ValueError('Empty temporal split')
    assert starts[train].max() + pd.Timedelta(hours=HORIZON-1) < starts[valid].min()
    return train, valid


def score_cube(truth, prediction):
    return float(roc_auc_score(truth.ravel(), prediction.ravel()))


def fit_member(spec, X, y, rounds, threads, valid=None):
    parameters = dict(spec['parameters'])
    if spec['backend'] == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(**parameters, n_estimators=int(rounds), n_jobs=threads,
                              **({'early_stopping_rounds': 80} if valid is not None else {}))
        model.fit(X, y, eval_set=[valid] if valid is not None else None, verbose=False)
        best = int(model.best_iteration) + 1 if valid is not None else int(rounds)
    else:
        import lightgbm as lgb
        parameters['num_threads'] = threads
        training = lgb.Dataset(X, label=y, free_raw_data=True)
        options = {}
        if valid is not None:
            parameters['metric'] = 'auc'
            options = dict(valid_sets=[lgb.Dataset(valid[0], label=valid[1], reference=training)],
                           callbacks=[lgb.early_stopping(80, verbose=False)])
        model = lgb.train(parameters, training, num_boost_round=int(rounds), **options)
        best = int(model.best_iteration or rounds)
    return model, best


class ChallengerRunner:
    def __init__(self, data, extra, local, additions, specs, threads, source_digest):
        self.data, self.local, self.additions = data, local, additions
        self.specs, self.threads = specs, min(4, int(threads))
        self.columns = list(data.feature_sets['full_engineered']) + list(extra.columns)
        self.base = np.column_stack([data.X_all[data.feature_sets['full_engineered']].to_numpy(np.float32),
                                     extra.to_numpy(np.float32)])
        self.raw = raw_digest(data.root)
        self.source_digest = source_digest
        self.directory = data.root / 'artifacts/challenger_v3'
        self.directory.mkdir(parents=True, exist_ok=True)

    def run_stage(self, member, train, predict, stage, rounds=None, tune=False):
        spec = self.specs[member]
        manifest = dict(member=member, spec=spec, train=list(map(int, train)),
                        predict=list(map(int, predict)), stage=stage, rounds=rounds,
                        tune=tune, source=self.source_digest, raw=self.raw, threads=self.threads,
                        packages={p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', spec['backend'])})
        signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
        directory = self.directory / member / signature
        directory.mkdir(parents=True, exist_ok=True)
        save_json(directory / 'manifest.json', manifest)
        cube = np.empty((len(predict), len(self.data.units), HORIZON), np.float32)
        selected, timings = {}, []
        train_rows, pred_rows = rows_for_cases(train), rows_for_cases(predict)
        for i, unit in enumerate(self.data.units):
            start = time.perf_counter()
            checkpoint = directory / f'unit_{i:02d}.npz'
            if checkpoint.exists():
                with np.load(checkpoint, allow_pickle=False) as saved:
                    values = saved['prediction'].copy()
                    best = int(saved['rounds'])
                status = 'restored'
            else:
                plant = unit.rsplit('_G', 1)[0]
                parts = [self.base, self.local[plant].to_numpy(np.float32)]
                names = self.columns + list(self.local[plant].columns)
                if spec['features'] == 'dispatch':
                    parts.append(self.additions[plant].to_numpy(np.float32))
                    names += list(self.additions[plant].columns)
                X = np.ascontiguousarray(np.column_stack(parts), dtype=np.float32)
                del parts
                y = self.data.target_cube[train, i].ravel()
                count = int(rounds[unit]) if rounds is not None else int(spec['max_rounds'])
                print(f'{stage} | {member} | {unit} | up to {count} rounds', flush=True)
                if len(np.unique(y)) == 1:
                    values = np.full((len(predict), HORIZON), y[0], np.float32)
                    best = 1
                else:
                    validation = (X[pred_rows], self.data.target_cube[predict, i].ravel()) if tune else None
                    # Limit BLAS as well as each model's own OpenMP threads.
                    with threadpool_limits(limits=self.threads), threadpool_limits(limits=1, user_api='blas'):
                        model, best = fit_member(spec, X[train_rows], y, count, self.threads, validation)
                        if spec['backend'] == 'xgboost':
                            values = model.predict_proba(X[pred_rows])[:, 1]
                            gain = model.feature_importances_
                        else:
                            values = model.predict(X[pred_rows], num_threads=self.threads)
                            gain = model.feature_importance(importance_type='gain')
                        values = values.reshape(len(predict), HORIZON).astype(np.float32)
                    pd.DataFrame({'feature': names, 'gain': gain}).sort_values('gain', ascending=False).to_csv(
                        directory / f'importance_{i:02d}.csv', index=False)
                    del model, validation
                temporary = checkpoint.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, prediction=values, rounds=best)
                os.replace(temporary, checkpoint)
                del X, y
                gc.collect()
                status = 'trained'
            if values.shape != (len(predict), HORIZON) or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                raise ValueError(f'Invalid predictions: {checkpoint}')
            cube[:, i] = values
            selected[unit] = int(best)
            seconds = time.perf_counter() - start
            timings.append(dict(unit=unit, rounds=best, seconds=seconds, status=status))
            pd.DataFrame(timings).to_csv(directory / 'runtime.csv', index=False)
            print(f'  {status}: {best} rounds, {seconds:.1f}s ({i+1}/{len(self.data.units)})', flush=True)
        return cube, selected

    def final_predictions(self, selection):
        train = np.arange(self.data.n_train_cases)
        test = np.arange(self.data.n_train_cases, self.data.n_all_cases)
        result = None
        for member, weight in selection['weights'].items():
            if weight <= 0:
                continue
            values, _ = self.run_stage(member, train, test, 'final', selection['rounds'][member])
            if result is None:
                result = np.zeros_like(values)
            result += float(weight) * values
        if result is None or abs(sum(selection['weights'].values()) - 1) > 1e-8:
            raise ValueError('Invalid ensemble weights')
        return np.clip(result, 0, 1)


def screen_challenger(runner, baseline_rounds):
    """2021 development, 2022 historical audit; final test labels are unavailable.

    These years have already informed earlier experiments, so the audit is not
    a new untouched test set. No per-unit weight search is performed.
    """
    predictions, all_rounds, rows = {}, {}, []
    for year in (2021, 2022):
        train, valid = purged_year(runner.data, year)
        predictions[year] = {}
        for member, spec in runner.specs.items():
            tuning = year == 2021 and spec['features'] == 'dispatch'
            counts = None if tuning else baseline_rounds.get(member, all_rounds.get(member))
            p, selected = runner.run_stage(member, train, valid, f'validation_{year}', counts, tune=tuning)
            all_rounds[member] = selected
            predictions[year][member] = p
            score = score_cube(runner.data.target_cube[valid], p)
            rows.append(dict(year=year, model=member, micro_auc=score))
            print(f'{year} | {member}: micro AUC={score:.6f}', flush=True)
        baseline = 0.5 * predictions[year]['xgb_v2'] + 0.5 * predictions[year]['lgb_v2']
        challenger = 0.5 * predictions[year]['xgb_dispatch'] + 0.5 * predictions[year]['lgb_dispatch']
        predictions[year]['baseline'] = baseline
        predictions[year]['challenger'] = challenger
        for label, p in [('baseline_50_50', baseline), ('dispatch_50_50', challenger)]:
            rows.append(dict(year=year, model=label, micro_auc=score_cube(runner.data.target_cube[valid], p)))
    scores = pd.DataFrame(rows)
    scores.to_csv(runner.directory / 'validation_scores.csv', index=False)
    _, development = purged_year(runner.data, 2021)
    _, audit = purged_year(runner.data, 2022)
    grid = []
    for alpha in (0., .25, .5, .75, 1.):
        p = (1-alpha)*predictions[2021]['baseline'] + alpha*predictions[2021]['challenger']
        grid.append((score_cube(runner.data.target_cube[development], p), alpha))
    # A tie favours the simpler existing baseline. Weights are global for micro AUC.
    dev_score, alpha = max(grid, key=lambda pair: (pair[0], -pair[1]))
    audited = (1-alpha)*predictions[2022]['baseline'] + alpha*predictions[2022]['challenger']
    audit_score = score_cube(runner.data.target_cube[audit], audited)
    baseline_score = score_cube(runner.data.target_cube[audit], predictions[2022]['baseline'])
    accepted = alpha > 0 and audit_score > baseline_score
    selected_alpha = alpha if accepted else 0.
    selection = dict(
        weights={'xgb_v2': (1-selected_alpha)/2, 'lgb_v2': (1-selected_alpha)/2,
                 'xgb_dispatch': selected_alpha/2, 'lgb_dispatch': selected_alpha/2},
        rounds=all_rounds, development_alpha=alpha, audit_candidate_auc=audit_score,
        audit_baseline_auc=baseline_score, passed_audit=bool(accepted),
        source_digest=runner.source_digest, raw_digest=runner.raw, specs=runner.specs)
    save_json(runner.directory / 'selection.json', selection)
    print('2021-selected challenger weight:', alpha, flush=True)
    print(f'2022 audit: candidate {audit_score:.6f}, baseline {baseline_score:.6f}', flush=True)
    print('Accepted new features.' if accepted else 'No confirmed improvement: guarded output uses baseline.', flush=True)
    # Diagnostics are descriptive only; these do not tune per-unit weights.
    details = []
    for i, unit in enumerate(runner.data.units):
        truth = runner.data.target_cube[audit, i].ravel()
        if len(np.unique(truth)) < 2:
            continue
        for name in ('baseline', 'challenger'):
            details.append(dict(unit=unit, model=name,
                                auc=float(roc_auc_score(truth, predictions[2022][name][:, i].ravel()))))
    pd.DataFrame(details).to_csv(runner.directory / 'audit_unit_scores.csv', index=False)
    return selection, scores
