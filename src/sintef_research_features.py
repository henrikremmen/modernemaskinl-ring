"""Deterministic input-only features for the next hydropower experiments.

No fitted transforms, labels, predicted/observed future storage, or solver outputs.
Water budgets deliberately exclude unknown dispatch and are explicitly proxies.
"""

import numpy as np
import pandas as pd
import yaml

from src.sintef_pipeline import HORIZON, read_time_csv


def window_sum(values, width, forward=True):
    """Truncated within-case sum, including the current hour; no edge repeats."""
    if not forward:
        return window_sum(values[:, ::-1], width, True)[:, ::-1]
    cumulative = np.pad(np.cumsum(values, axis=1, dtype=np.float64), ((0, 0), (1, 0)))
    left = np.arange(values.shape[1])
    right = np.minimum(left + width, values.shape[1])
    return cumulative[:, right] - cumulative[:, left]


def consecutive_true(mask):
    """Run length ending/starting at each position, reset at each case boundary."""
    past = np.zeros(mask.shape, dtype=np.float32)
    future = np.zeros_like(past)
    for h in range(mask.shape[1]):
        past[:, h] = mask[:, h] * (1 + (past[:, h - 1] if h else 0))
    for h in range(mask.shape[1] - 1, -1, -1):
        future[:, h] = mask[:, h] * (1 + (future[:, h + 1] if h + 1 < mask.shape[1] else 0))
    return past, future


def plant_sources(topology, plant):
    """Follow reservoir -> plant or reservoir -> tunnel -> plant connections.

    Stop at the first reservoir; do not pretend all upstream water arrives instantly.
    """
    connections = topology['connections']
    found = set()

    def visit(name, kind, seen):
        if (name, kind) in seen:
            return
        seen = seen | {(name, kind)}
        for edge in connections:
            if edge['to'] != name or edge['to_type'] != kind:
                continue
            if edge['from_type'] == 'reservoir':
                found.add(edge['from'])
            elif edge['from_type'] == 'tunnel':
                visit(edge['from'], 'tunnel', seen)

    visit(plant, 'plant', set())
    if not found:
        raise ValueError(f'No source reservoir found for {plant}')
    return sorted(found)


def build_research_features(data):
    """Return global additions, per-plant additions, and an auditable source mapping."""
    root = data.root
    with open(root / 'data/extended/Tokke_Vinje_topology.yaml') as stream:
        topology = yaml.safe_load(stream)
    starts = pd.DatetimeIndex(pd.concat([data.train_cases, data.test_cases])['starttime'])
    n = data.n_all_cases
    horizon = np.arange(HORIZON)
    time_index = starts.repeat(HORIZON) + pd.to_timedelta(np.tile(horizon, n), unit='h')
    price = data.X_all['price'].to_numpy().reshape(n, HORIZON)
    volume = read_time_csv(root / 'data/kernel/Historical_volume_2015_2024.csv').set_index('date').reindex(starts)
    water = read_time_csv(root / 'data/kernel/Synthetic_water_value_2015_2024.csv').set_index('date').reindex(starts + pd.Timedelta(days=7))
    inflow = read_time_csv(root / 'data/kernel/Historical_inflow_1958_2025.csv').set_index('date').reindex(time_index)
    minimum = read_time_csv(root / 'data/extended/Constraint_min_volume.csv').set_index('date').reindex(time_index)
    extra = {}

    def add(target, name, values):
        values = np.broadcast_to(values, (n, HORIZON)).reshape(-1).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite research feature {name}')
        target[name] = values

    add(extra, 'v2_day_in_horizon', np.broadcast_to(horizon // 24, (n, HORIZON)))
    add(extra, 'v2_hours_remaining', HORIZON - horizon)
    day_price = price.reshape(n, 7, 24)
    day_mean = np.repeat(day_price.mean(axis=2), 24, axis=1)
    day_rank = pd.DataFrame(day_price.reshape(-1, 24)).rank(axis=1, pct=True).to_numpy().reshape(n, HORIZON)
    add(extra, 'v2_price_day_rank', day_rank)
    add(extra, 'v2_price_minus_day_mean', price - day_mean)
    prefix_mean = np.cumsum(price, axis=1) / (horizon + 1)
    suffix_mean = np.cumsum(price[:, ::-1], axis=1)[:, ::-1] / (HORIZON - horizon)
    add(extra, 'v2_price_minus_prefix_mean', price - prefix_mean)
    add(extra, 'v2_price_minus_suffix_mean', price - suffix_mean)
    add(extra, 'v2_best_remaining_price_gap', np.maximum.accumulate(price[:, ::-1], axis=1)[:, ::-1] - price)
    reservoirs = topology['model']['reservoir']
    reservoir_inputs = {}
    for name, config in reservoirs.items():
        cap = float(config['max_vol'])
        initial = volume[name].to_numpy()[:, None]
        terminal = water[name].to_numpy()[:, None]
        flow = inflow[name].to_numpy().reshape(n, HORIZON)
        # m3/s * 3600 / 1e6 -> million m3 per hour, matching storage units.
        hourly_volume = flow * 0.0036
        cumulative = np.cumsum(hourly_volume, axis=1)
        remaining = np.cumsum(hourly_volume[:, ::-1], axis=1)[:, ::-1]
        floor = minimum[name].to_numpy().reshape(n, HORIZON) if name in minimum else np.zeros_like(price)
        budget = initial + cumulative
        add(extra, f'v2_cumulative_inflow_fill__{name}', cumulative / cap)
        add(extra, f'v2_remaining_inflow_fill__{name}', remaining / cap)
        add(extra, f'v2_no_release_excess__{name}', np.maximum(budget - cap, 0) / cap)
        add(extra, f'v2_no_release_min_slack__{name}', (budget - floor) / cap)
        curve = config['vol_head']
        head = np.interp(initial[:, 0], curve['x'], curve['y'])[:, None]
        reservoir_inputs[name] = (cap, initial, terminal, cumulative, remaining, head)

    plants = {}
    mappings = []
    for plant, plant_config in topology['model']['plant'].items():
        sources = plant_sources(topology, plant)
        values = [reservoir_inputs[s] for s in sources]
        capacity = sum(v[0] for v in values)
        initial = sum(v[1] for v in values)
        # Multi-reservoir summaries are capacity-weighted approximations, not SHOP prices.
        terminal = sum(v[2] * v[0] for v in values) / capacity
        head = sum(v[5] * v[0] for v in values) / capacity - float(plant_config['outlet_line'])
        cum = sum(v[3] for v in values)
        remain = sum(v[4] for v in values)
        margin = price - terminal
        out = {}
        add(out, 'local_initial_fill', initial / capacity)
        add(out, 'local_gross_head_proxy', head)
        add(out, 'local_terminal_water_value', terminal)
        add(out, 'local_margin', margin)
        add(out, 'local_margin_normalized', margin / (np.abs(terminal) + price.std(axis=1)[:, None] + 1))
        add(out, 'local_cumulative_inflow_fill', cum / capacity)
        add(out, 'local_remaining_inflow_fill', remain / capacity)
        add(out, 'local_margin_x_fill', margin * initial / capacity)
        positive = np.maximum(margin, 0)
        negative = np.maximum(-margin, 0)
        past, future = consecutive_true(margin > 0)
        add(out, 'local_profitable_run_past', past)
        add(out, 'local_profitable_run_future', future)
        add(out, 'local_profitable_fraction_week', (margin > 0).mean(axis=1)[:, None])
        for hours in (6, 24):
            add(out, f'local_positive_margin_next_{hours}h', window_sum(positive, hours))
            add(out, f'local_negative_margin_next_{hours}h', window_sum(negative, hours))
            add(out, f'local_positive_margin_past_{hours}h', window_sum(positive, hours, False))
        # Start cost / rated production gives EUR/MWh; margin sums are hourly
        # EUR/MWh contributions. This is an approximate rated-power payback ratio.
        unit_configs = [topology['model']['generator'][u] for u in data.units if u.rsplit('_G', 1)[0] == plant]
        cost_per_mw = np.mean([(v.get('startcost_const', 0) + v.get('stopcost_const', 0)) / v['p_max'] for v in unit_configs])
        add(out, 'local_start_stop_payback_proxy', window_sum(positive, 24) / max(cost_per_mw, 1e-6))
        plants[plant] = pd.DataFrame(out)
        mappings.append({'plant': plant, 'source_reservoirs': '|'.join(sources), 'source_capacity_Mm3': capacity})
    return pd.DataFrame(extra), plants, pd.DataFrame(mappings)
