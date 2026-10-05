"""Exact feature arithmetic tests; no estimator is fitted."""
import numpy as np
from src.sintef_feature_refinement import price_competition, _hours_to_future_max


def test_competition_matches_explicit_within_case_windows():
    price = np.array([[4, 2, 4, -1, 9], [-3, -3, -3, -3, -3]], dtype=float)
    actual = price_competition(price, widths=(3, 5, 8))
    for row, values in enumerate(price):
        for width in (3, 5, 8):
            for t, value in enumerate(values):
                for direction, others in [('future', values[t+1:t+width]),
                                          ('past', values[max(0,t-width+1):t])]:
                    count = np.mean(others > value) if len(others) else 0
                    premium = np.maximum(others-value,0).mean()/(values.std()+1) if len(others) else 0
                    np.testing.assert_allclose(actual[f'v4_better_{direction}_fraction_{width}'][row,t],count,atol=1e-7)
                    np.testing.assert_allclose(actual[f'v4_{direction}_premium_{width}'][row,t],premium,atol=1e-7)


def test_future_max_uses_nearest_tie_and_resets_cases():
    values = np.array([[2,5,1,5,0],[9,2,2,1,1]],dtype=float)
    np.testing.assert_array_equal(_hours_to_future_max(values),[[1,0,1,0,0],[0,0,0,0,0]])
