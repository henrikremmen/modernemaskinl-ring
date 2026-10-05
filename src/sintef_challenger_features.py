"""Input-only storage-network and commitment features; never fits a model.

The dispatch calculation is a two-state economic proxy, not a SHOP simulation.
It uses the supplied 168-hour price/inflow path, initial volume and terminal
water values. No future observed volumes or commitment labels are consulted.
"""
import numpy as np
import pandas as pd
import yaml
from src.sintef_pipeline import HORIZON, read_time_csv
from src.sintef_research_features import plant_sources, window_sum


def commitment_advantage(reward, start_cost, stop_cost):
    """Best objective with hour ON minus best objective with hour OFF.

    Exhaustive two-state dynamic program with a free initial and terminal state.
    Reward and costs have the same units. Input shape is cases x horizon.
    """
    reward = np.asarray(reward, dtype=np.float64)
    n, h = reward.shape
    f0 = np.zeros((n, h)); f1 = np.zeros_like(f0)
    a = np.zeros(n); b = np.zeros(n)
    for t in range(h):
        a, b = np.maximum(a, b - stop_cost), np.maximum(b, a - start_cost) + reward[:, t]
        f0[:, t], f1[:, t] = a, b
    a = np.zeros(n); b = np.zeros(n)
    result = np.empty_like(reward)
    for t in range(h - 1, -1, -1):
        result[:, t] = f1[:, t] + b - f0[:, t] - a
        a, b = (np.maximum(a, b + reward[:, t] - start_cost),
                np.maximum(a - stop_cost, b + reward[:, t]))
    return result.astype(np.float32)


def tunnel_storage_group(topology, sources):
    """Include upstream reservoirs connected without passing through a turbine.

    This storage group is an availability proxy: tunnel limits still matter.
    """
    found = set(sources)
    pending = [(s, 'reservoir') for s in sources]
    seen = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        for edge in topology['connections']:
            if (edge['to'], edge['to_type']) != node:
                continue
            if edge['from_type'] == 'tunnel':
                pending.append((edge['from'], 'tunnel'))
            elif edge['from_type'] == 'reservoir' and node[1] == 'tunnel':
                found.add(edge['from'])
                pending.append((edge['from'], 'reservoir'))
    return sorted(found)


def build_challenger_features(data):
    with open(data.root / 'data/extended/Tokke_Vinje_topology.yaml') as stream:
        topology = yaml.safe_load(stream)
    n = data.n_all_cases
    h = np.arange(HORIZON)
    starts = pd.DatetimeIndex(pd.concat([data.train_cases, data.test_cases])['starttime'])
    times = starts.repeat(HORIZON) + pd.to_timedelta(np.tile(h, n), unit='h')
    price = data.X_all['price'].to_numpy().reshape(n, HORIZON)
    volume = read_time_csv(data.root / 'data/kernel/Historical_volume_2015_2024.csv').set_index('date').reindex(starts)
    water = read_time_csv(data.root / 'data/kernel/Synthetic_water_value_2015_2024.csv').set_index('date').reindex(starts + pd.Timedelta(days=7))
    inflow = read_time_csv(data.root / 'data/kernel/Historical_inflow_1958_2025.csv').set_index('date').reindex(times)
    floors = read_time_csv(data.root / 'data/extended/Constraint_min_volume.csv').set_index('date').reindex(times)
    min_flow = read_time_csv(data.root / 'data/extended/Constraint_min_flow.csv').set_index('date').reindex(times)
    reservoirs = topology['model']['reservoir']
    plants, audit = {}, []
    for plant, cfg in topology['model']['plant'].items():
        sources = plant_sources(topology, plant)
        group = tunnel_storage_group(topology, sources)
        # Broader passive catchment, kept distinct from tunnel-connected storage.
        # In particular Langeidvatn feeds Vatjern/Haukeli through a river, not a tunnel.
        seen, pending = set(), [(s, 'reservoir') for s in sources]
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend((e['from'], e['from_type']) for e in topology['connections']
                           if (e['to'], e['to_type']) == node
                           and e['from_type'] in ('reservoir', 'river', 'tunnel'))
        catchment = sorted(name for name, kind in seen if kind == 'reservoir')
        rivers = sorted(name for name, kind in seen if kind == 'river')
        downstream = sorted({e['to'] for e in topology['connections']
                             if e['from'] == plant and e['from_type'] == 'plant'
                             and e['to_type'] == 'reservoir'})
        unit_configs = [v for u, v in topology['model']['generator'].items()
                        if u.rsplit('_G', 1)[0] == plant]
        pmax = sum(float(v['p_max']) for v in unit_configs)
        pmin_fraction = sum(float(v['p_min']) for v in unit_configs) / pmax
        startup = sum(float(v['startcost_const']) for v in unit_configs) / pmax
        shutdown = sum(float(v['stopcost_const']) for v in unit_configs) / pmax
        source_capacity = sum(float(reservoirs[s]['max_vol']) for s in sources)
        cap = sum(float(reservoirs[s]['max_vol']) for s in group)
        initial = volume[group].sum(axis=1).to_numpy()[:, None]
        source_initial = volume[sources].sum(axis=1).to_numpy()[:, None]
        flow = inflow[group].sum(axis=1).to_numpy().reshape(n, HORIZON)
        minimum = sum((floors[s].to_numpy().reshape(n, HORIZON) for s in group if s in floors), np.zeros_like(price))
        terminal = sum(water[s].to_numpy()[:, None] * reservoirs[s]['max_vol'] for s in sources) / source_capacity
        head = sum(np.interp(volume[s], reservoirs[s]['vol_head']['x'], reservoirs[s]['vol_head']['y'])[:, None]
                   * reservoirs[s]['max_vol'] for s in sources) / source_capacity - float(cfg['outlet_line'])
        # p[MW] = 0.00981 * head[m] * flow[m3/s] * efficiency.
        # 90% efficiency is an explicit approximation, not a measured turbine curve.
        rated_flow = pmax / (0.00981 * np.maximum(head, 1.) * 0.90)
        hourly_release = rated_flow * 0.0036
        incoming = flow * 0.0036
        cumulative = np.cumsum(incoming, axis=1)
        remaining = np.cumsum(incoming[:, ::-1], axis=1)[:, ::-1]
        available = np.maximum(initial + incoming.sum(axis=1)[:, None] - minimum.max(axis=1)[:, None], 0)
        hours_budget = np.clip(available / hourly_release, 0, HORIZON)
        required_hours = np.clip((initial + incoming.sum(axis=1)[:, None] - cap) / hourly_release, 0, HORIZON)
        sorted_price = np.sort(price, axis=1)[:, ::-1]
        cutoff_index = np.clip(np.ceil(hours_budget[:, 0]).astype(int) - 1, 0, HORIZON - 1)
        scarcity_price = sorted_price[np.arange(n), cutoff_index][:, None]
        out = {}

        def add(name, values):
            values = np.broadcast_to(values, (n, HORIZON)).ravel().astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite feature {plant}/{name}')
            out['v3_' + name] = values

        add('connected_fill', initial / cap)
        add('connected_minus_local_fill', initial / cap - source_initial / source_capacity)
        add('connected_slack', (initial + cumulative - minimum) / cap)
        add('connected_remaining_inflow', remaining / cap)
        add('connected_inflow_to_rated_flow', flow / rated_flow)
        add('connected_available_full_load_hours', hours_budget)
        add('connected_mandatory_full_load_hours', required_hours)
        add('price_minus_storage_budget_cutoff', price - scarcity_price)
        add('storage_budget_minus_water_value', scarcity_price - terminal)
        add('head', head)
        add('initial_hours_at_rated_flow', initial / hourly_release)
        add('price_times_head', price * head)
        add('no_release_spill_fill', np.maximum(initial + cumulative - cap, 0) / cap)
        catchment_capacity = sum(float(reservoirs[s]['max_vol']) for s in catchment)
        catchment_volume = volume[catchment].sum(axis=1).to_numpy()[:, None]
        catchment_flow_columns = catchment + [r for r in rivers if r in inflow.columns]
        catchment_flow = inflow[catchment_flow_columns].sum(axis=1).to_numpy().reshape(n, HORIZON)
        catchment_water = sum(water[s].to_numpy()[:, None] * reservoirs[s]['max_vol'] for s in catchment) / catchment_capacity
        mandatory_flow = sum((min_flow[r].to_numpy().reshape(n, HORIZON) for r in rivers if r in min_flow.columns), np.zeros_like(price))
        add('catchment_fill', catchment_volume / catchment_capacity)
        add('catchment_inflow_to_rated', catchment_flow / rated_flow)
        add('catchment_inflow_week_to_rated', catchment_flow.mean(axis=1)[:, None] / rated_flow)
        add('catchment_remaining_inflow_fill', np.cumsum(catchment_flow[:, ::-1] * .0036, axis=1)[:, ::-1] / catchment_capacity)
        add('catchment_minus_local_water', catchment_water - terminal)
        add('catchment_mandatory_flow_to_rated', mandatory_flow / rated_flow)
        add('catchment_initial_full_load_hours', catchment_volume / hourly_release)
        add('catchment_no_release_excess', np.maximum(catchment_volume + np.cumsum(catchment_flow*.0036, axis=1) - catchment_capacity, 0) / catchment_capacity)
        for reservoir in downstream:
            add('downstream_fill_' + reservoir, volume[reservoir].to_numpy()[:, None] / reservoirs[reservoir]['max_vol'])
            add('local_minus_downstream_water_' + reservoir, terminal - water[reservoir].to_numpy()[:, None])
        margin = price - terminal
        # A local EUR/MWh water value is already relative to the downstream plant;
        # do not incorrectly subtract a downstream EUR/MWh value a second time.
        for threshold_name, threshold in [('water', terminal), ('budget', np.maximum(terminal, scarcity_price))]:
            gain = price - threshold
            reward = np.where(gain >= 0, gain, gain * pmin_fraction)
            for scale in (0.5, 1.0, 2.0):
                advantage = commitment_advantage(reward, startup * scale, shutdown * scale)
                add(f'dp_{threshold_name}_{scale:g}', advantage)
                add(f'dp_{threshold_name}_{scale:g}_on', advantage > 0)
            dispatch_proxy = np.maximum(gain, 0) > 0
            release = np.cumsum(dispatch_proxy * hourly_release, axis=1)
            add(f'proxy_storage_{threshold_name}', (initial + cumulative - release) / cap)
        for width in (3, 12, 48, 96, 168):
            positive = window_sum(np.maximum(margin, 0), width)
            negative = window_sum(np.maximum(-margin, 0), width)
            add(f'profit_next_{width}', positive)
            add(f'loss_next_{width}', negative)
            add(f'profit_past_{width}', window_sum(np.maximum(margin, 0), width, False))
        days = price.reshape(n, 7, 24)
        for statistic, values in [('min', days.min(axis=2)), ('max', days.max(axis=2)), ('std', days.std(axis=2))]:
            add('daily_' + statistic, np.repeat(values, 24, axis=1))
        for q in (.1, .25, .5, .75, .9):
            add(f'margin_quantile_{q}', np.quantile(price, q, axis=1)[:, None] - terminal)
        plants[plant] = pd.DataFrame(out)
        audit.append(dict(plant=plant, direct_sources='|'.join(sources), connected_storage='|'.join(group), passive_catchment='|'.join(catchment),
                          downstream='|'.join(downstream), total_MW=pmax, start_EUR_per_MW=startup))
    return plants, pd.DataFrame(audit)
