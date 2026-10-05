"""Resumable, CPU-budgeted experiments; importing this module never trains models."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
import platform
import time

import numpy as np
import pandas as pd

from src.sintef_pipeline import HORIZON, RANDOM_SEED, rows_for_cases, micro_auc, make_submission


def atomic_npz(path, **arrays):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def atomic_json(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp.replace(path)


def input_fingerprint(root):
    """Hash raw inputs, never old prediction artifacts or notebook outputs."""
    paths = sorted((root / 'data').rglob('*.csv')) + sorted((root / 'data').rglob('*.yaml'))
    paths += [root / 'prediction_mapping.csv', root / 'sample_submission.csv']
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def inner_time_split(data, outer_train, days=120):
    starts = pd.DatetimeIndex(data.train_cases['starttime'])
    boundary = starts[outer_train].max().normalize() - pd.Timedelta(days=days - 1)
    train = outer_train[starts[outer_train] + pd.Timedelta(hours=HORIZON - 1) < boundary]
    stop = outer_train[starts[outer_train] >= boundary]
    if len(train) < 365 or len(stop) < 60:
        raise ValueError('Insufficient history for purged inner stopping split')
    assert starts[train].max() + pd.Timedelta(hours=HORIZON - 1) < starts[stop].min()
    return train, stop


def experiment_tasks(data, mode):
    if mode == 'unit':
        return [(u, u.rsplit('_G', 1)[0], [i]) for i, u in enumerate(data.units)]
    if mode != 'plant_state':
        raise ValueError(mode)
    plants = list(dict.fromkeys(u.rsplit('_G', 1)[0] for u in data.units))
    return [(p, p, [i for i, u in enumerate(data.units) if u.rsplit('_G', 1)[0] == p]) for p in plants]


def encode_states(cube):
    """case x unit x hour -> case-hour integer bit pattern."""
    return (cube.transpose(0, 2, 1).reshape(-1, cube.shape[1]) @ (1 << np.arange(cube.shape[1]))).astype(np.int32)


def state_marginals(probabilities, n_units):
    probabilities = np.asarray(probabilities)
    if n_units == 1:
        return probabilities.reshape(-1, 1)
    states = np.arange(1 << n_units)
    bits = ((states[:, None] >> np.arange(n_units)) & 1).astype(np.float64)
    return probabilities @ bits


def fit_tree(backend, X, labels, n_units, config, rounds, stopping=None):
    """Early stopping sees inner history only. Outer labels never enter this function."""
    threads = int(config['threads_per_model'])
    if backend == 'xgboost':
        from xgboost import XGBClassifier
        if n_units != 1:
            raise ValueError('XGBoost experiment uses separate binary models')
        parameters = dict(
            objective='binary:logistic', eval_metric='auc', n_estimators=rounds,
            learning_rate=0.03, max_depth=7, min_child_weight=50,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1,
            reg_lambda=5.0, tree_method='hist', max_bin=256,
            device='cpu', n_jobs=threads, random_state=RANDOM_SEED,
        )
        if stopping is not None:
            parameters['early_stopping_rounds'] = config['patience']
        model = XGBClassifier(**parameters)
        model.fit(X, labels, eval_set=[stopping] if stopping is not None else None, verbose=False)
        best = int(model.best_iteration) + 1 if stopping is not None else rounds
        return model, best

    import lightgbm as lgb
    parameters = dict(
        objective='binary' if n_units == 1 else 'multiclass', metric='None',
        learning_rate=0.03, num_leaves=31, max_depth=-1,
        min_data_in_leaf=150, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=1, lambda_l1=0.1,
        lambda_l2=5.0, max_bin=255, num_threads=threads,
        device_type='cpu', deterministic=True, force_col_wise=True,
        seed=RANDOM_SEED, verbosity=-1,
    )
    if n_units > 1:
        parameters['num_class'] = 1 << n_units
    train_set = lgb.Dataset(X, label=labels, free_raw_data=True)
    kwargs = {}
    if stopping is not None:
        X_stop, y_stop = stopping
        stop_set = lgb.Dataset(X_stop, label=y_stop, reference=train_set)

        def unit_auc(prediction, dataset):
            from sklearn.metrics import roc_auc_score
            truth = ((dataset.get_label().astype(int)[:, None] >> np.arange(n_units)) & 1)
            pred = state_marginals(prediction, n_units)
            return 'unit_micro_auc', float(roc_auc_score(truth.ravel(), pred.ravel())), True

        kwargs = dict(valid_sets=[stop_set], feval=unit_auc,
                      callbacks=[lgb.early_stopping(config['patience'], first_metric_only=True, verbose=False)])
    model = lgb.train(parameters, train_set, num_boost_round=rounds, **kwargs)
    return model, int(model.best_iteration or rounds)


def predict_tree(model, backend, X, n_units):
    if backend == 'xgboost':
        return model.predict_proba(X)[:, 1, None]
    return state_marginals(model.predict(X), n_units)


class ResearchExperiment:
    """At most outer_jobs concurrent fits, each with threads_per_model threads.

    Completed fold/unit predictions and stopping rounds survive kernel restarts.
    An interrupted individual fit starts again; this is not mid-tree resumption.
    """

    def __init__(self, data, extra, local, config, code_fingerprint, raw_fingerprint):
        self.data, self.local = data, local
        self.config = dict(config)
        self.tasks = experiment_tasks(data, config['mode'])
        columns = data.feature_sets['full_engineered']
        self.base = data.X_all[columns].to_numpy(dtype=np.float32)
        self.names = list(columns)
        if config['features'] == 'physics':
            self.base = np.column_stack([self.base, extra.to_numpy(dtype=np.float32)])
            self.names += list(extra.columns)
        self.base = np.ascontiguousarray(self.base)
        versions = {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', config['backend']]}
        manifest = dict(config=self.config, code=code_fingerprint, raw_data=raw_fingerprint,
                        packages=versions, python=platform.python_version(),
                        units=data.units, base_features=self.names,
                        local_features={p: list(v.columns) for p, v in local.items()},
                        validation='outer 2020/2021/2022; purged inner 120-day stopping; refit outer history')
        self.signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        self.directory = data.root / 'artifacts/research_runs' / config['name'] / self.signature[:16]
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_json(self.directory / 'manifest.json', manifest)
        budget = config['outer_jobs'] * config['threads_per_model']
        if budget > (os.cpu_count() or 1):
            raise ValueError('CPU budget exceeds logical CPU count')
        print(f"{config['name']}: {len(self.tasks)} tasks per fold; {config['outer_jobs']} fits x {config['threads_per_model']} threads", flush=True)
        print('Checkpoint directory:', self.directory)

    def matrix(self, plant):
        if self.config['features'] == 'physics':
            return np.ascontiguousarray(np.column_stack([self.base, self.local[plant].to_numpy(dtype=np.float32)]))
        return self.base

    def labels(self, cases, units):
        return encode_states(self.data.target_cube[np.asarray(cases)][:, units, :])

    def run_task(self, task_index, train_cases, predict_cases, stage, rounds=None):
        name, plant, units = self.tasks[task_index]
        folder = self.directory / stage
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'task_{task_index:02d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if not np.array_equal(cached['case_ids'], predict_cases):
                    raise ValueError(f'Checkpoint case IDs changed: {path}')
                prediction = cached['prediction'].copy()
                best = int(cached['rounds'])
            print(f'{stage} | {name} | restored', flush=True)
            return units, prediction, best
        start = time.perf_counter()
        print(f'{stage} | {name} | starting', flush=True)
        X = self.matrix(plant)
        y = self.labels(train_cases, units)
        backend = self.config['backend']
        count = len(units)
        if rounds is None:
            round_path = folder / f'task_{task_index:02d}_rounds.json'
            if round_path.exists():
                rounds = int(json.loads(round_path.read_text())['rounds'])
            else:
                inner, stop = inner_time_split(self.data, train_cases)
                y_inner, y_stop = self.labels(inner, units), self.labels(stop, units)
                if len(np.unique(y_inner)) < 2 or len(np.unique(y_stop)) < 2:
                    rounds = 300
                    print(f'{name}: constant inner labels; fixed 300-round fallback', flush=True)
                else:
                    tuner, rounds = fit_tree(
                        backend, X[rows_for_cases(inner)], y_inner, count,
                        self.config, self.config['max_rounds'],
                        stopping=(X[rows_for_cases(stop)], y_stop),
                    )
                    del tuner
                atomic_json(round_path, {'rounds': int(rounds)})
        print(f'{stage} | {name} | refit on training history: {rounds} rounds', flush=True)
        if len(np.unique(y)) == 1:
            bits = ((int(y[0]) >> np.arange(count)) & 1).astype(np.float32)
            prediction = np.broadcast_to(bits[None, :, None], (len(predict_cases), count, HORIZON)).copy()
            gain = np.zeros(X.shape[1], dtype=np.float32)
        else:
            model, _ = fit_tree(backend, X[rows_for_cases(train_cases)], y, count, self.config, int(rounds))
            values = predict_tree(model, backend, X[rows_for_cases(predict_cases)], count)
            prediction = values.reshape(len(predict_cases), HORIZON, count).transpose(0, 2, 1).astype(np.float32)
            gain = model.feature_importances_ if backend == 'xgboost' else model.feature_importance(importance_type='gain')
            if stage == 'final':
                suffix = '.ubj' if backend == 'xgboost' else '.txt'
                model.save_model(str(folder / f'model_{task_index:02d}{suffix}'))
            del model
        if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
            raise ValueError(f'Invalid predictions for {name}')
        atomic_npz(path, prediction=prediction, rounds=np.asarray(rounds),
                   case_ids=predict_cases, importance=gain)
        print(f'{stage} | {name} | {rounds} rounds | {time.perf_counter()-start:.1f}s', flush=True)
        return units, prediction, int(rounds)

    def parallel_tasks(self, train, predict, stage, rounds=None):
        # BLAS limits are set once around the pool, never concurrently per worker.
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api='blas'):
            with ThreadPoolExecutor(max_workers=self.config['outer_jobs']) as pool:
                futures = [pool.submit(self.run_task, i, train, predict, stage,
                                       None if rounds is None else rounds[i]) for i in range(len(self.tasks))]
                return [f.result() for f in futures]

    def evaluate(self):
        oof = np.full(self.data.target_cube.shape, np.nan, dtype=np.float32)
        rows, unit_rows, rounds = [], [], []
        from sklearn.metrics import roc_auc_score
        for fold in self.data.folds:
            ids = fold['validation_cases']
            results = self.parallel_tasks(fold['train_cases'], ids, f"fold_{fold['year']}")
            prediction = np.empty((len(ids), len(self.data.units), HORIZON), dtype=np.float32)
            for units, values, _ in results:
                prediction[:, units, :] = values
            oof[ids] = prediction
            rounds.append([r[2] for r in results])
            score = micro_auc(self.data.target_cube[ids], prediction)
            rows.append({'validation_year': fold['year'], 'micro_auc': score})
            for u, name in enumerate(self.data.units):
                unit_rows.append(dict(validation_year=fold['year'], unit=name,
                                      auc=roc_auc_score(self.data.target_cube[ids, u].ravel(), prediction[:, u].ravel())))
            print(f"Outer {fold['year']}: micro AUC={score:.6f}", flush=True)
        atomic_npz(self.directory / 'oof.npz', oof=oof, rounds=np.asarray(rounds))
        scores = pd.DataFrame(rows)
        scores.to_csv(self.directory / 'fold_scores.csv', index=False)
        pd.DataFrame(unit_rows).to_csv(self.directory / 'unit_scores.csv', index=False)
        return scores

    def importance_table(self):
        rows = []
        for fold in self.data.folds:
            for i, (name, plant, _) in enumerate(self.tasks):
                path = self.directory / f"fold_{fold['year']}" / f'task_{i:02d}.npz'
                with np.load(path, allow_pickle=False) as cached:
                    values = cached['importance'].astype(float)
                columns = self.names + (list(self.local[plant].columns) if self.config['features'] == 'physics' else [])
                values /= max(values.sum(), 1e-12)
                rows.extend({'validation_year': fold['year'], 'task': name,
                             'feature': feature, 'normalized_gain': float(value)}
                            for feature, value in zip(columns, values))
        result = pd.DataFrame(rows)
        result.to_csv(self.directory / 'feature_importance.csv', index=False)
        return result.groupby('feature').normalized_gain.mean().sort_values(ascending=False)

    def finalize(self):
        """Load OOF/rounds from disk, so final training works after a restart."""
        with np.load(self.directory / 'oof.npz', allow_pickle=False) as saved:
            oof = saved['oof'].copy()
            rounds = np.maximum(1, np.median(saved['rounds'], axis=0).astype(int))
        prediction_dir = self.data.root / 'artifacts/predictions'
        prediction_dir.mkdir(parents=True, exist_ok=True)
        tag = self.config['name']
        export_meta = prediction_dir / f'{tag}_manifest.json'
        submission = self.data.root / 'submissions' / f'{tag}.csv'
        if export_meta.exists():
            if json.loads(export_meta.read_text())['signature'] != self.signature:
                raise ValueError('This export name belongs to a different run. Change config name to preserve it.')
        elif submission.exists():
            raise ValueError('Existing submission without matching manifest. Choose a new experiment name.')
        atomic_json(export_meta, {'signature': self.signature, 'run_directory': str(self.directory),
                                  'config': self.config, 'complete': False})
        train = np.arange(self.data.n_train_cases)
        test = np.arange(self.data.n_train_cases, self.data.n_all_cases)
        results = self.parallel_tasks(train, test, 'final', rounds=rounds)
        cube = np.empty((len(test), len(self.data.units), HORIZON), dtype=np.float32)
        for units, prediction, _ in results:
            cube[:, units, :] = prediction
        atomic_npz(prediction_dir / f'{tag}_oof.npz', oof=oof)
        pd.read_csv(self.directory / 'fold_scores.csv').to_csv(prediction_dir / f'{tag}_fold_scores.csv', index=False)
        path, _ = make_submission(self.data, cube, f'{tag}.csv')
        atomic_json(export_meta, {'signature': self.signature, 'run_directory': str(self.directory),
                                  'config': self.config, 'complete': True})
        print('Saved:', path, flush=True)
        return path


def compare_research_runs(runs, legacy='xgboost_per_generator'):
    """Read-only OOF comparison. Old models used outer-year early stopping."""
    rows = []
    for run in runs:
        scores = pd.read_csv(run.directory / 'fold_scores.csv')
        rows.append(dict(model=run.config['name'], mean_auc=scores.micro_auc.mean(),
                         worst_auc=scores.micro_auc.min(), protocol='inner stopping'))
    old = runs[0].data.root / 'artifacts/predictions' / f'{legacy}_fold_scores.csv'
    if old.exists():
        scores = pd.read_csv(old)
        rows.append(dict(model=legacy, mean_auc=scores.micro_auc.mean(),
                         worst_auc=scores.micro_auc.min(), protocol='legacy outer stopping'))
    return pd.DataFrame(rows).sort_values('mean_auc', ascending=False)
