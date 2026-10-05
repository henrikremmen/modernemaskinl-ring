"""Workflow checks with a fake predictor: deliberately no model training."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import src.sintef_challenger_runner as runner_module


def test_predictions_resume_and_changed_config_invalidates_cache(tmp_path, monkeypatch):
    calls = []

    class FakePredictor:
        feature_importances_ = np.ones(3)

        def predict_proba(self, X):
            return np.tile([.7, .3], (len(X), 1))

    def fake_fit(spec, X, y, rounds, threads, valid):
        calls.append((X.shape, len(y), rounds, threads, valid))
        return FakePredictor(), rounds

    monkeypatch.setattr(runner_module, 'fit_member', fake_fit)
    monkeypatch.setattr(runner_module, 'raw_digest', lambda root: 'test-raw')
    monkeypatch.setattr(runner_module.importlib.metadata, 'version', lambda p: 'test')
    data = SimpleNamespace(root=tmp_path, units=['Plant_G1'], n_train_cases=2, n_all_cases=3,
                           feature_sets={'full_engineered': ['price']},
                           X_all=pd.DataFrame({'price': np.arange(3*168, dtype=np.float32)}),
                           target_cube=np.tile([0, 1], 168).reshape(2, 1, 168))
    extra = pd.DataFrame({'extra': np.zeros(3*168, np.float32)})
    local = {'Plant': pd.DataFrame({'local': np.zeros(3*168, np.float32)})}
    specs = {'test': dict(backend='xgboost', features='physics', parameters={}, max_rounds=10)}
    runner = runner_module.ChallengerRunner(data, extra, local, {}, specs, 4, 'test-source')
    p, counts = runner.run_stage('test', np.array([0]), np.array([2]), 'final', {'Plant_G1': 7})
    np.testing.assert_allclose(p, .3)
    assert p.shape == (1, 1, 168)
    assert counts == {'Plant_G1': 7}
    assert calls[0][:4] == ((168, 3), 168, 7, 4)
    assert calls[0][4] is None  # Never try to access test labels.
    runner.run_stage('test', np.array([0]), np.array([2]), 'final', {'Plant_G1': 7})
    assert len(calls) == 1
    runner.run_stage('test', np.array([0]), np.array([2]), 'final', {'Plant_G1': 8})
    assert len(calls) == 2
