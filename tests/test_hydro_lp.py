"""LP relaxation, runner and export checks; no real model training."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.special import expit
from src.sintef_hydro_lp import HydroLP, lp_plant_features, solve_all_cases, PLANT_SOURCE
from src.sintef_lp_experiment import (LPRunner, select_candidate, block_bootstrap_delta, verify_submission,
                                      MAX_DEV_YEAR_LOSS)

PROJECT = Path(__file__).resolve().parents[1]
T = 168
TOPOLOGY = yaml.safe_load((PROJECT / 'data/extended/Tokke_Vinje_topology.yaml').read_text())
MIN_FLOW = ['b_Kjelavatn', 'b_Hyljelihyl', 'b_Vinjevatn', 'b_Byrtevatn', 'b_Bandak', 'r_Vest_Vassdraget']
MIN_VOL = ['Totak', 'Byrtevatn', 'Staavatn']


def synthetic_case(lp, price=50., wv=40., r_vest_min=0.):
    res = lp.res
    return dict(price=np.full(T, price) if np.isscalar(price) else price,
                inflow={r: np.zeros(T) for r in res + ['r_Vest_Vassdraget']},
                v0={r: .5 * TOPOLOGY['model']['reservoir'][r]['max_vol'] for r in res},
                wv_end={r: wv for r in res},
                min_flow={r: np.full(T, r_vest_min if r == 'r_Vest_Vassdraget' else 0.) for r in MIN_FLOW},
                min_volume={r: np.zeros(T) for r in MIN_VOL})


def plant(lp, out, name, k):
    return out[lp.plants.index(name), k]


def test_lp_economics_coupling_and_duals():
    lp = HydroLP(TOPOLOGY, MIN_FLOW, MIN_VOL)
    idle = lp.solve(**synthetic_case(lp, price=50., wv=5000.))
    assert np.allclose(idle[:, 0], 0, atol=1e-9)           # water worth more than any hour
    assert (idle[:, 2] < 0).all()                         # every plant wants less flow
    busy = lp.solve(**synthetic_case(lp, price=50., wv=40.))
    assert plant(lp, busy, 'Byrte', 0).min() > .999       # big reservoirs: run whenever price > value
    assert plant(lp, busy, 'Tokke', 0).mean() > .99
    # Vinje (165 m3/s) feeds 9.36-Mm3 Vinjevatn drained by Tokke (128 m3/s): coupling binds.
    assert plant(lp, busy, 'Vinje', 0).mean() < .95
    # Unconstrained Byrte: dual threshold equals its local terminal water value.
    np.testing.assert_allclose(plant(lp, busy, 'Byrte', 1), 40., atol=1e-4)


def test_lp_minimum_flow_forces_lio_below_water_value():
    lp = HydroLP(TOPOLOGY, MIN_FLOW, MIN_VOL)
    free = lp.solve(**synthetic_case(lp, price=50., wv=1000., r_vest_min=0.))
    forced = lp.solve(**synthetic_case(lp, price=50., wv=1000., r_vest_min=10.))
    assert plant(lp, free, 'Lio', 0).max() < 1e-9
    np.testing.assert_allclose(plant(lp, forced, 'Lio', 0), 10 / 14, atol=1e-6)  # 10 of 14 m3/s


def test_real_cases_are_finite_and_thread_invariant():
    starts = pd.to_datetime(['2021-03-01', '2022-07-15', '2024-12-25'], utc=True)
    one, plants, _ = solve_all_cases(PROJECT, starts, threads=1)
    two, _, _ = solve_all_cases(PROJECT, starts, threads=2)
    np.testing.assert_array_equal(one, two)
    assert one.shape == (3, 9, 6, T) and np.isfinite(one).all()
    assert ((one[:, :, 0] > -1e-6) & (one[:, :, 0] < 1 + 1e-6)).all()
    assert plants == list(TOPOLOGY['model']['plant'])


def test_plant_features_shapes_and_ranges():
    rng = np.random.default_rng(0)
    plants = ['Tokke', 'Lio']
    cube = rng.uniform(size=(4, 2, 6, T)).astype(np.float32)
    price = rng.normal(50, 10, size=(4, T))
    frames = lp_plant_features(cube, plants, price, {p: np.full(4, 40.) for p in plants}, {'Tokke': 4, 'Lio': 1})
    assert len(frames['Lio'].columns) == 12 and len(frames['Tokke'].columns) == 13
    for frame in frames.values():
        assert frame.shape[0] == 4 * T and np.isfinite(frame.to_numpy()).all()
        for c in ('lp_next_24h_fraction', 'lp_past_24h_fraction', 'lp_day_fraction', 'lp_week_fraction'):
            assert frame[c].between(0, 1).all()
    q = cube[:, 1, 0]
    np.testing.assert_allclose(frames['Lio']['lp_next_24h_fraction'].to_numpy().reshape(4, T)[:, -1], q[:, -1], rtol=1e-6)


class FakeModel:
    def __init__(self, width):
        self.feature_importances_ = np.ones(width)

    def predict_proba(self, X, iteration_range):
        p = expit(X[:, 0] / 10 + X[:, -1] + iteration_range[1] / 100)
        return np.column_stack([1 - p, p])

    def predict(self, X, num_iteration, num_threads):
        return self.predict_proba(X, (0, num_iteration))[:, 1]

    def feature_importance(self, importance_type):
        return self.feature_importances_


def toy(tmp_path):
    units = ['A_G1', 'A_G2']
    target = np.zeros((5, 2, T), np.int8)
    target[:, 0, ::2] = 1
    columns = ['Run No'] + [f'result_committed_{u}_t{h}' for u in units for h in range(T)]
    sample = pd.DataFrame(np.zeros((2, len(columns))), columns=columns)
    sample['Run No'] = [5, 6]
    data = SimpleNamespace(root=tmp_path, units=units, n_train_cases=5, n_all_cases=7, n_test_cases=2,
                           target_cube=target, X_all=pd.DataFrame({'price': np.arange(7 * T, dtype=np.float32) / 100}),
                           feature_sets={'full_engineered': ['price']}, test_cases=pd.DataFrame({'Run No': [5, 6]}),
                           sample_submission=sample)
    sample.to_csv(tmp_path / 'sample_submission.csv', index=False)
    frame = lambda name, v: {'A': pd.DataFrame({name: np.full(7 * T, v, np.float32)})}
    parts = dict(extra=pd.DataFrame({'global': np.zeros(7 * T, np.float32)}), local=frame('local', 0.),
                 competition=frame('competition', 1.), paths=frame('path', 2.), lp=frame('lp_x', 3.))
    specs = {f: dict(backend=b, parameters={}) for f, b in [('xgb', 'xgboost'), ('lgb', 'lightgbm')]}
    rounds = {f: dict(zip(units, [2, 4])) for f in specs}
    return data, parts, specs, rounds


def make_runner(tmp_path, monkeypatch, module='src.sintef_lp_experiment', use_checkpoints=True, ns=None):
    import importlib
    data, parts, specs, rounds = toy(tmp_path)
    if ns is None:
        mod = importlib.import_module(module)
        monkeypatch.setattr(mod, 'raw_digest', lambda root: 'raw')
        cls = mod.LPRunner
    else:
        cls = ns['LPRunner']
    return cls(data, parts['extra'], parts['local'], parts['competition'], parts['paths'], parts['lp'],
               specs, rounds, 1, 'source', tmp_path / 'artifacts/hydro_lp_v6', use_checkpoints)


def test_runner_resume_variant_isolation_and_guards(tmp_path, monkeypatch):
    import src.sintef_lp_experiment as module
    calls = []

    def fake_fit(spec, X, y, rounds, threads):
        calls.append((X.shape[1], rounds))
        return FakeModel(X.shape[1]), rounds
    monkeypatch.setattr(module, 'fit_member', fake_fit)
    runner = make_runner(tmp_path, monkeypatch)
    control = runner.run('xgb', 'v5', [0, 1], [2, 3], 'dev')
    candidate = runner.run('xgb', 'v5_lp', [0, 1], [2, 3], 'dev')
    assert calls == [(5, 4), (6, 4)]  # 2 inherited trees x 2; constant second unit needs no estimator; LP adds one column
    assert not np.array_equal(control['2'], candidate['2'])
    again = runner.run('xgb', 'v5_lp', [0, 1], [2, 3], 'dev')
    assert len(calls) == 2
    np.testing.assert_array_equal(again['1.5'], candidate['1.5'])
    np.testing.assert_array_equal(candidate['2'][:, 1], 0)
    assert (runner.directory / 'runtime_log.csv').exists()
    with pytest.raises(ValueError, match='overlapping'):
        runner.run('xgb', 'v5', [0, 1], [1, 2], 'bad')
    with pytest.raises(ValueError, match='Unknown variant'):
        runner.run('xgb', 'lp_only', [0, 1], [2, 3], 'bad')
    runner.lp['A'] = pd.DataFrame({'path': np.zeros(7 * T, np.float32)})
    with pytest.raises(ValueError, match='Duplicate'):
        runner.run('xgb', 'v5_lp', [0, 1], [2, 3], 'dup')


def test_selection_rule_is_preregistered(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, monkeypatch)
    identity = json.loads(json.dumps(runner.identity()))

    def write(deltas):
        for year, d in deltas.items():
            (runner.directory / f'summary_development_{year}.json').write_text(json.dumps(dict(delta=d, identity=identity)))
        return select_candidate(runner, list(deltas))
    assert write({2020: .0004, 2021: .0001})['accepted']
    assert not write({2020: .002, 2021: -MAX_DEV_YEAR_LOSS - 1e-6})['accepted']
    assert not write({2020: -.0001, 2021: .00005})['accepted']
    assert write({2020: .0002, 2021: -.0001})['variant'] == 'v5_lp'


def test_block_bootstrap_and_submission_checks(tmp_path):
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=(30, 2, T)) > .5).astype(int)
    p = rng.uniform(size=y.shape)
    interval, share = block_bootstrap_delta(y, p, p, reps=20)
    assert np.allclose(interval, 0) and share == 0
    columns = ['Run No', 'a', 'b']
    sample = pd.DataFrame({'Run No': [1, 2], 'a': [0., 0.], 'b': [0., 0.]})
    sample.to_csv(tmp_path / 's.csv', index=False)
    sample.assign(a=[.2, .3]).to_csv(tmp_path / 'ok.csv', index=False)
    assert verify_submission(tmp_path / 'ok.csv', tmp_path / 's.csv') == (2, 3)
    sample.iloc[::-1].to_csv(tmp_path / 'bad.csv', index=False)
    with pytest.raises(AssertionError):
        verify_submission(tmp_path / 'bad.csv', tmp_path / 's.csv')
    sample[columns[::-1]].to_csv(tmp_path / 'bad2.csv', index=False)
    with pytest.raises(AssertionError):
        verify_submission(tmp_path / 'bad2.csv', tmp_path / 's.csv')


def test_short_notebook_is_self_contained_cold_and_runs_final(tmp_path, monkeypatch):
    short = json.loads((PROJECT / 'artifacts/hydro_lp_v6/Short_notebook_hydro_lp_v6.ipynb').read_text())
    research = json.loads((PROJECT / 'notebooks/19_hydro_lp_relaxation.ipynb').read_text())
    for nb in (short, research):
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                ast.parse(''.join(c['source']))
        code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        assert 'from src.' not in code and 'import src' not in code
        assert 'total_calculation_time' not in code.replace("'total_calculation_time'", '')  # never a feature
    short_code = '\n'.join(''.join(c['source']) for c in short['cells'] if c['cell_type'] == 'code')
    assert "RUN_PROFILE = 'professor'" in short_code
    assert 'def evaluate_year(' not in short_code and 'def select_candidate(' not in short_code
    ns = {}
    for c in short['cells']:
        if c.get('id') in ('resources', 'pipeline', 'physics', 'fit_helpers', 'competition', 'paths', 'staged',
                           'hydro_lp', 'experiment'):
            exec(compile(''.join(c['source']), c['id'], 'exec'), ns)
    assert ns['THREADS'] <= 4 and ns['USE_CHECKPOINTS'] is False
    ns['raw_digest'] = lambda root: 'raw'
    ns['fit_member'] = lambda spec, X, y, rounds, threads: (FakeModel(X.shape[1]), rounds)
    runner = make_runner(tmp_path, monkeypatch, ns=ns)
    path, frame = ns['final_lp'](runner)
    assert path.exists() and frame.shape == (2, 337)
    assert frame.columns.tolist() == runner.data.sample_submission.columns.tolist()
    assert (tmp_path / 'submissions/xgb_hydro_lp_v6.csv').exists()


def test_evaluate_select_final_pipeline_on_toy_years(tmp_path, monkeypatch):
    import src.sintef_lp_experiment as module
    monkeypatch.setattr(module, 'raw_digest', lambda root: 'raw')
    monkeypatch.setattr(module, 'fit_member', lambda spec, X, y, rounds, threads: (FakeModel(X.shape[1]), rounds))
    rng = np.random.default_rng(3)
    n_train, units = 30, ['A_G1', 'A_G2']
    starts = pd.date_range('2020-12-10', periods=n_train, freq='D', tz='UTC')
    target = (rng.uniform(size=(n_train, 2, T)) > .5).astype(np.int8)
    columns = ['Run No'] + [f'result_committed_{u}_t{h}' for u in units for h in range(T)]
    sample = pd.DataFrame(np.zeros((2, len(columns))), columns=columns)
    sample['Run No'] = [n_train, n_train + 1]
    sample.to_csv(tmp_path / 'sample_submission.csv', index=False)
    n = n_train + 2
    data = SimpleNamespace(root=tmp_path, units=units, n_train_cases=n_train, n_all_cases=n, n_test_cases=2,
                           target_cube=target, train_cases=pd.DataFrame({'starttime': starts}),
                           X_all=pd.DataFrame({'price': rng.normal(size=n * T).astype(np.float32)}),
                           feature_sets={'full_engineered': ['price']}, test_cases=pd.DataFrame({'Run No': [n_train, n_train + 1]}),
                           sample_submission=sample)
    frame = lambda name: {'A': pd.DataFrame({name: rng.normal(size=n * T).astype(np.float32)})}
    specs = {f: dict(backend=b, parameters={}) for f, b in [('xgb', 'xgboost'), ('lgb', 'lightgbm')]}
    runner = module.LPRunner(data, pd.DataFrame({'g': np.zeros(n * T, np.float32)}), frame('local'), frame('comp'),
                             frame('path'), frame('lp_x'), specs, {f: {u: 3 for u in units} for f in specs}, 1, 's',
                             tmp_path / 'art')
    summary, table, diagnostics = module.evaluate_year(runner, 2021, 'development')
    assert set(table.model) >= {'blend_v5', 'blend_v5_lp', 'xgb_v5', 'lgb_v5_lp'}
    assert {'unit', 'start_month', 'horizon_hours', 'utc_hour', 'all'} <= set(diagnostics.group)
    assert np.isfinite(summary['delta']) and len(summary['bootstrap_delta_p05_p50_p95']) == 3
    selection = module.select_candidate(runner, [2021])
    assert selection['variant'] == ('v5_lp' if summary['delta'] > 0 else 'v5')
    path, submission = module.final_lp(runner)
    assert submission.shape == (2, 337) and path.name == 'hydro_lp_v6.csv'


def test_verdict_respects_uncertainty_and_identity_guard(tmp_path, monkeypatch):
    from src.sintef_lp_experiment import verdict, load_checked
    audit = lambda d, lo, hi: dict(delta=d, bootstrap_delta_p05_p50_p95=[lo, d, hi])
    accepted = dict(accepted=True)
    assert verdict(accepted, audit(1e-5, -8e-4, 9e-4))[0] == 'POSITIVE BUT UNCERTAIN'
    assert verdict(accepted, audit(5e-4, 1e-4, 9e-4))[0] == 'SUPPORTED'
    assert verdict(accepted, audit(-1e-4, -5e-4, 2e-4))[0] == 'NOT CONFIRMED'
    assert verdict(dict(accepted=False), audit(5e-4, 1e-4, 9e-4))[0] == 'NOT SUPPORTED'
    runner = make_runner(tmp_path, monkeypatch)
    (runner.directory / 'selection.json').write_text(json.dumps(dict(accepted=True, identity=runner.identity())))
    assert load_checked(runner, 'selection.json')['accepted']
    runner.source_digest = 'edited code'
    with pytest.raises(ValueError, match='another'):
        load_checked(runner, 'selection.json')
    with pytest.raises(FileNotFoundError):
        load_checked(runner, 'summary_audit_2022.json')
    research = json.loads((PROJECT / 'notebooks/19_hydro_lp_relaxation.ipynb').read_text())
    ids = [c.get('id') for c in research['cells']]
    assert ids.index('conclusion') < ids.index('final')
    final = ''.join(research['cells'][ids.index('final')]['source'])
    assert "load_checked(runner, 'selection.json')" in final and 'final_manifest.json' in final
