"""Pure feature/protocol tests. These tests never fit a model or execute a notebook."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.sintef_research_features import window_sum, consecutive_true, plant_sources
from src.sintef_research_runner import (
    atomic_npz, encode_states, state_marginals, inner_time_split, experiment_tasks,
)


class ResearchHelperTests(unittest.TestCase):
    def test_windows_do_not_cross_cases_or_repeat_edges(self):
        x = np.array([[1, 2, 3], [10, 20, 30]])
        np.testing.assert_array_equal(window_sum(x, 2), [[3, 5, 3], [30, 50, 30]])
        np.testing.assert_array_equal(window_sum(x, 2, False), [[1, 3, 5], [10, 30, 50]])

    def test_profitable_runs_reset_at_boundaries(self):
        past, future = consecutive_true(np.array([[0, 1, 1], [1, 0, 1]], dtype=bool))
        np.testing.assert_array_equal(past, [[0, 1, 2], [1, 0, 1]])
        np.testing.assert_array_equal(future, [[0, 2, 1], [1, 0, 1]])

    def test_state_roundtrip_and_generator_order(self):
        bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1)
        cube = bits.T[None, :, :]
        np.testing.assert_array_equal(encode_states(cube), np.arange(16))
        np.testing.assert_array_equal(state_marginals(np.eye(16), 4), bits)
        np.testing.assert_array_equal(state_marginals(np.array([.1, .9]), 1), [[.1], [.9]])

    def test_joint_probabilities_are_marginalized_not_argmax(self):
        probabilities = np.array([[.1, .2, .3, .4]])
        np.testing.assert_allclose(state_marginals(probabilities, 2), [[.6, .7]])

    def test_purged_inner_split_is_before_outer_validation(self):
        starts = pd.date_range('2015-01-01', periods=1500, tz='UTC')
        data = SimpleNamespace(train_cases=pd.DataFrame({'starttime': starts}))
        train, stop = inner_time_split(data, np.arange(1000))
        self.assertEqual(len(stop), 120)
        self.assertLess(starts[train].max() + pd.Timedelta(hours=167), starts[stop].min())
        self.assertLessEqual(stop.max(), 999)
        self.assertEqual(len(set(train) & set(stop)), 0)

    def test_tunnel_source_mapping_stops_at_reservoir(self):
        def edge(a, b, at, bt):
            return dict(**{'from': a, 'to': b}, from_type=at, to_type=bt)
        topology = {'connections': [edge('r1', 't', 'reservoir', 'tunnel'),
                                    edge('t', 'p', 'tunnel', 'plant'),
                                    edge('r2', 'p', 'reservoir', 'plant'),
                                    edge('g', 'p', 'generator', 'plant'),
                                    edge('r3', 'r2', 'reservoir', 'reservoir')]}
        self.assertEqual(plant_sources(topology, 'p'), ['r1', 'r2'])

    def test_all_units_covered_once(self):
        data = SimpleNamespace(units=['A_G2', 'B_G1', 'A_G1'])
        tasks = experiment_tasks(data, 'plant_state')
        self.assertEqual(tasks, [('A', 'A', [0, 2]), ('B', 'B', [1])])

    def test_atomic_checkpoint_has_no_object_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'task.npz'
            atomic_npz(path, prediction=np.array([.2, .8]), rounds=np.asarray(31))
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(int(saved['rounds']), 31)
                np.testing.assert_array_equal(saved['prediction'], [.2, .8])
            self.assertFalse(path.with_name('task.npz.tmp').exists())


if __name__ == '__main__':
    unittest.main()
