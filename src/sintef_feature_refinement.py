"""Feature-only refinements for the frozen notebook 12/13 estimators."""
import numpy as np
import pandas as pd
import yaml
from src.sintef_pipeline import HORIZON, read_time_csv
from src.sintef_research_features import plant_sources, window_sum
from src.sintef_challenger_runner import purged_year, score_cube, save_json


def price_competition(price, widths=(24, 72, 168)):
    """Strictly better alternative hours within this case, excluding current hour.

    Compute relative position and mean forgone price premium separately before
    and after the current hour. No edge replication or cross-case windows.
    """
    price = np.asarray(price, dtype=np.float64)
    n, horizon = price.shape
    ahead_count = np.zeros_like(price); past_count = np.zeros_like(price)
    ahead_gain = np.zeros_like(price); past_gain = np.zeros_like(price)
    scale = price.std(axis=1)[:, None] + 1.
    output = {}
    for distance in range(1, max(widths)):
        if distance < horizon:
            gap = price[:, distance:] - price[:, :-distance]
            ahead_count[:, :-distance] += gap > 0
            past_count[:, distance:] += gap < 0
            ahead_gain[:, :-distance] += np.maximum(gap, 0)
            past_gain[:, distance:] += np.maximum(-gap, 0)
        width = distance + 1
        if width in widths:
            ahead_n = np.maximum(1, np.minimum(width-1, horizon-1-np.arange(horizon)))
            past_n = np.maximum(1, np.minimum(width-1, np.arange(horizon)))
            output[f'v4_better_future_fraction_{width}'] = ahead_count / ahead_n
            output[f'v4_better_past_fraction_{width}'] = past_count / past_n
            output[f'v4_future_premium_{width}'] = ahead_gain / ahead_n / scale
            output[f'v4_past_premium_{width}'] = past_gain / past_n / scale
    return {k: v.astype(np.float32) for k, v in output.items()}


def build_refinement_features(data):
    n = data.n_all_cases
    price = data.X_all['price'].to_numpy().reshape(n, HORIZON)
    common = pd.DataFrame({k: v.ravel() for k, v in price_competition(price).items()})
    starts = pd.DatetimeIndex(pd.concat([data.train_cases, data.test_cases])['starttime'])
    times = starts.repeat(HORIZON) + pd.to_timedelta(np.tile(np.arange(HORIZON), n), unit='h')
    with open(data.root / 'data/extended/Tokke_Vinje_topology.yaml') as stream:
        topology = yaml.safe_load(stream)
    volumes = read_time_csv(data.root / 'data/kernel/Historical_volume_2015_2024.csv').set_index('date').reindex(starts)
    water = read_time_csv(data.root / 'data/kernel/Synthetic_water_value_2015_2024.csv').set_index('date').reindex(starts + pd.Timedelta(days=7))
    inflow = read_time_csv(data.root / 'data/kernel/Historical_inflow_1958_2025.csv').set_index('date').reindex(times)
    minimum = read_time_csv(data.root / 'data/extended/Constraint_min_volume.csv').set_index('date').reindex(times)
    price_only, combined, audit = {}, {}, []
    for plant in topology['model']['plant']:
        sources = plant_sources(topology, plant)
        cap = sum(topology['model']['reservoir'][s]['max_vol'] for s in sources)
        terminal = sum(water[s].to_numpy()[:, None] * topology['model']['reservoir'][s]['max_vol'] for s in sources) / cap
        margin = price - terminal
        flow = inflow[sources].sum(axis=1).to_numpy().reshape(n, HORIZON)
        initial = volumes[sources].sum(axis=1).to_numpy()[:, None]
        floor = sum((minimum[s].to_numpy().reshape(n, HORIZON) for s in sources if s in minimum), np.zeros_like(price))
        out = {}

        def add(name, values):
            values = np.broadcast_to(values, (n, HORIZON)).ravel().astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite feature {plant}/{name}')
            out['v4_' + name] = values

        prefix = np.arange(1, HORIZON+1)
        suffix = prefix[::-1]
        positive = margin > 0
        add('profitable_prefix_fraction', np.cumsum(positive, axis=1) / prefix)
        add('profitable_suffix_fraction', np.cumsum(positive[:, ::-1], axis=1)[:, ::-1] / suffix)
        inflow_cumulative = np.cumsum(flow, axis=1)
        flow_total = flow.sum(axis=1)[:, None]
        fraction_arrived = np.divide(inflow_cumulative, flow_total,
                                     out=np.zeros_like(inflow_cumulative), where=flow_total > 0)
        profit = np.maximum(margin, 0)
        profit_total = profit.sum(axis=1)[:, None]
        fraction_profit_elapsed = np.divide(np.cumsum(profit, axis=1), profit_total,
                                            out=np.zeros_like(profit), where=profit_total > 0)
        add('inflow_arrival_fraction', fraction_arrived)
        add('profit_elapsed_fraction', fraction_profit_elapsed)
        add('inflow_minus_profit_timing', fraction_arrived - fraction_profit_elapsed)
        add('no_profitable_hours', profit_total == 0)
        add('no_inflow', flow_total == 0)
        for width in (24, 72, 168):
            flow_window = window_sum(flow, width)
            weighted = np.divide(window_sum(flow*price, width), flow_window,
                                 out=np.zeros_like(price, dtype=np.float64), where=flow_window > 0)
            add(f'inflow_weighted_price_gap_{width}', np.where(flow_window > 0, weighted-price, 0))
            add(f'profitable_inflow_fill_{width}', window_sum(flow*positive, width)*.0036/cap)
        future_floor = np.maximum.accumulate(floor[:, ::-1], axis=1)[:, ::-1]
        add('upcoming_floor_increase_fill', (future_floor-floor)/cap)
        add('initial_slack_to_future_floor', (initial-future_floor)/cap)
        add('hours_until_highest_remaining_price', _hours_to_future_max(price))
        add('profitable_hours_remaining', np.cumsum(positive[:, ::-1], axis=1)[:, ::-1])
        water_features = pd.DataFrame(out)
        price_only[plant] = common
        combined[plant] = pd.concat([common, water_features], axis=1)
        audit.append(dict(plant=plant, sources='|'.join(sources),
                          price_features=len(common.columns), timing_features=len(water_features.columns)))
    return {'competition': price_only, 'competition_timing': combined}, pd.DataFrame(audit)


def _hours_to_future_max(price):
    n, horizon = price.shape
    result = np.zeros_like(price)
    best = np.full(n, -np.inf)
    index = np.full(n, horizon-1)
    for t in range(horizon-1, -1, -1):
        update = price[:, t] >= best
        best = np.maximum(best, price[:, t])
        index = np.where(update, t, index)
        result[:, t] = index-t
    return result


def screen_refinements(runner, fixed_rounds):
    """Controlled ablation: same per-unit rounds/settings for all feature variants.

    Choose the best new feature variant per backend on 2021, then audit the
    resulting fixed 50/50 blend on 2022. These years have been reused historically.
    """
    rows, dev_scores = [], {}
    train, valid = purged_year(runner.data, 2021)
    for member, spec in runner.specs.items():
        key = 'xgb_v2' if spec['backend'] == 'xgboost' else 'lgb_v2'
        p, _ = runner.run_stage(member, train, valid, 'ablation_2021', fixed_rounds[key])
        score = score_cube(runner.data.target_cube[valid], p)
        dev_scores[member] = score
        rows.append(dict(year=2021, model=member, micro_auc=score))
        print(f'2021 | {member}: {score:.6f}', flush=True)
    chosen = [max((m for m,s in runner.specs.items() if s['backend']==backend and s['features']!='physics'),
                  key=lambda m: dev_scores[m]) for backend in ('xgboost','lightgbm')]
    train, valid = purged_year(runner.data, 2022)
    results = {}
    for member in ['xgb_v2','lgb_v2'] + chosen:
        key = 'xgb_v2' if runner.specs[member]['backend']=='xgboost' else 'lgb_v2'
        results[member], _ = runner.run_stage(member, train, valid, 'audit_2022', fixed_rounds[key])
        rows.append(dict(year=2022, model=member, micro_auc=score_cube(runner.data.target_cube[valid],results[member])))
    new = (results[chosen[0]] + results[chosen[1]])*.5
    old = (results['xgb_v2']+results['lgb_v2'])*.5
    new_score, old_score = (score_cube(runner.data.target_cube[valid],p) for p in (new,old))
    rows += [dict(year=2022,model='new_blend',micro_auc=new_score),
             dict(year=2022,model='baseline_blend',micro_auc=old_score)]
    selection = dict(weights={m:.5 for m in chosen},
                     rounds={m:fixed_rounds['xgb_v2' if runner.specs[m]['backend']=='xgboost' else 'lgb_v2'] for m in chosen},
                     source_digest=runner.source_digest, raw_digest=runner.raw, specs=runner.specs,
                     audit_new_auc=new_score, audit_old_auc=old_score, passed_audit=bool(new_score>old_score))
    table = pd.DataFrame(rows)
    table.to_csv(runner.directory/'ablation_scores.csv',index=False)
    save_json(runner.directory/'selection.json',selection)
    print(f'2022 blend: new {new_score:.6f}, baseline {old_score:.6f}',flush=True)
    if not selection['passed_audit']:
        print('New features did not improve the historical audit. CSVs remain experimental.',flush=True)
    return selection,table
