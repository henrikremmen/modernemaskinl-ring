"""Analytical checks only: no model training."""
import itertools
import numpy as np
from src.sintef_challenger_features import commitment_advantage, tunnel_storage_group


def test_dynamic_program_matches_exhaustive_enumeration():
    rewards = np.array([[3., -1., -5., 4., 2.], [-2., -1., 2., -2., 1.]])
    start, stop = 2.1, 1.3
    actual = commitment_advantage(rewards, start, stop)
    for k, reward in enumerate(rewards):
        best = np.full((len(reward), 2), -np.inf)
        for sequence in itertools.product((0, 1), repeat=len(reward)):
            sequence = np.array(sequence)
            profit = np.dot(sequence, reward)
            profit -= start * np.sum(np.diff(sequence) == 1)
            profit -= stop * np.sum(np.diff(sequence) == -1)
            for h, bit in enumerate(sequence):
                best[h, bit] = max(best[h, bit], profit)
        np.testing.assert_allclose(actual[k], best[:, 1]-best[:, 0], atol=1e-6)


def test_tunnel_group_stops_at_turbines_and_handles_cycles():
    def edge(a, b, at, bt):
        return dict(**{'from': a, 'to': b}, from_type=at, to_type=bt)
    topology = {'connections': [edge('up', 'tunnel', 'reservoir', 'tunnel'),
        edge('tunnel', 'down', 'tunnel', 'reservoir'),
        edge('upstream_plant', 'up', 'plant', 'reservoir'),
        edge('remote', 'upstream_plant', 'reservoir', 'plant'),
        edge('down', 'back', 'reservoir', 'tunnel'),
        edge('back', 'up', 'tunnel', 'reservoir')]}
    assert tunnel_storage_group(topology, ['down']) == ['down', 'up']


def test_zero_switch_cost_recovers_hourly_margins():
    reward = np.array([[1, -4, 3, 0, 5]], dtype=float)
    np.testing.assert_allclose(commitment_advantage(reward, 0, 0), reward)
