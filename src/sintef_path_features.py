"""Fixed, input-only whole-week descriptors for notebook 18.

No labels, fitted transforms, future observed storage, or cross-case windows.
The supplied future price/inflow paths are inputs to each scheduling problem.
"""
import numpy as np
import pandas as pd
import yaml
from scipy.fft import dct
from src.sintef_pipeline import HORIZON
from src.sintef_research_features import plant_sources, window_sum


def contiguous_competition(price):
    """Consecutive strictly better hours immediately before/after each hour.

    Unlike a rank, an interruption ends the sequence. Current hour is excluded.
    Equal prices end a run. Each case is independent.
    """
    price = np.asarray(price)
    n, h = price.shape
    result = {}
    for label, values in [('next', price), ('past', price[:, ::-1])]:
        active = np.ones((n, h), dtype=bool)
        count = np.zeros((n, h), np.float32)
        premium = np.zeros((n, h), np.float32)
        for shift in range(1, h):
            gap = values[:, shift:] - values[:, :-shift]
            active[:, :-shift] &= gap > 0
            count[:, :-shift] += active[:, :-shift]
            premium[:, :-shift] += np.maximum(gap, 0) * active[:, :-shift]
        result[f'path_consecutive_better_{label}'] = count if label=='next' else count[:, ::-1]
        result[f'path_contiguous_premium_{label}'] = premium if label=='next' else premium[:, ::-1]
    return result


def weekly_shape(price):
    """Daily and low-frequency shape, preserving absolute day positions in a case."""
    price = np.asarray(price, dtype=np.float64)
    n, h = price.shape
    if h != HORIZON:
        raise ValueError('Expected exactly 168 hourly prices')
    scale = price.std(axis=1)[:, None] + 1.
    centered = (price-price.mean(axis=1)[:, None])/scale
    out = {}
    daily = price.reshape(n, 7, 24)
    for day in range(7):
        # Compare current hour with EACH day, not just its own day or whole week.
        out[f'path_day{day}_mean_gap'] = (daily[:, day].mean(axis=1)[:, None]-price)/scale
        out[f'path_day{day}_peak_gap'] = (daily[:, day].max(axis=1)[:, None]-price)/scale
    coefficients = dct(centered, type=2, norm='ortho', axis=1)
    for k in range(1, 7):
        out[f'path_price_dct_{k}'] = np.broadcast_to(coefficients[:, k, None], (n,h))
    for forward, label in [(True,'next'), (False,'past')]:
        for width in (48, 96, 168):
            counts = np.minimum(width, np.arange(h,0,-1) if forward else np.arange(1,h+1))
            average = window_sum(price,width,forward)/counts
            out[f'path_price_{label}_{width}_relative'] = (price-average)/scale
    for name, values in contiguous_competition(price).items():
        out[name] = values/scale if 'premium' in name else values
    return {name: np.asarray(value, dtype=np.float32) for name,value in out.items()}


def build_path_features(data, local):
    n = data.n_all_cases
    price = data.X_all['price'].to_numpy().reshape(n,HORIZON)
    global_frame = pd.DataFrame({k:v.ravel() for k,v in weekly_shape(price).items()})
    with open(data.root/'data/extended/Tokke_Vinje_topology.yaml') as stream:
        topology = yaml.safe_load(stream)
    frames, records = {}, []
    for plant in local:
        sources = plant_sources(topology,plant)
        cap = sum(topology['model']['reservoir'][r]['max_vol'] for r in sources)
        terminal = local[plant]['local_terminal_water_value'].to_numpy().reshape(n,HORIZON)[:,0,None]
        margin = price-terminal
        scale = price.std(axis=1)[:,None]+np.abs(terminal)+1.
        daily_margin = margin.reshape(n,7,24)
        # All inputs come from X_all, already restricted to the supplied case window.
        flow = sum(data.X_all[f'inflow__{r}'].to_numpy().reshape(n,HORIZON) for r in sources)
        daily_flow = flow.reshape(n,7,24)
        extra = {}
        def add(name, value):
            extra['path_'+name] = np.broadcast_to(value,(n,HORIZON)).ravel().astype(np.float32)
        for day in range(7):
            add(f'day{day}_positive_margin', np.maximum(daily_margin[:,day],0).sum(axis=1)[:,None]/scale)
            add(f'day{day}_negative_margin', np.maximum(-daily_margin[:,day],0).sum(axis=1)[:,None]/scale)
            add(f'day{day}_source_inflow_fill',daily_flow[:,day].sum(axis=1)[:,None]*.0036/cap)
        profit=np.maximum(margin,0)
        profit_sum=profit.sum(axis=1)[:,None]
        shares=np.divide(profit,profit_sum,out=np.zeros_like(profit),where=profit_sum>0)
        add('profit_concentration', (shares**2).sum(axis=1)[:,None])
        add('profit_center_of_time', (shares*np.arange(HORIZON)).sum(axis=1)[:,None]/(HORIZON-1))
        add('current_profit_share',shares)
        frame=pd.concat([global_frame,pd.DataFrame(extra)],axis=1)
        if not np.isfinite(frame.to_numpy()).all() or not frame.columns.is_unique:
            raise ValueError(f'Invalid path features for {plant}')
        frames[plant]=frame
        records.append(dict(plant=plant,sources='|'.join(sources),global_features=len(global_frame.columns),local_features=len(extra)))
    return frames,pd.DataFrame(records)
