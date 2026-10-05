"""Weekly linear relaxation of the Tokke-Vinje watercourse; input-only features, never fits a model.

One LP per scheduling case: maximise price revenue plus SHOP-style end value of stored water,
subject to reservoir balances, plant/tunnel/river limits, minimum flows and soft minimum volumes.
Uses only the case's supplied price/inflow path, initial volume, terminal water value, constraint
series and topology. No labels, no SHOP results, no observed storage after case start.
Deliberate simplifications (the boosted model corrects them): linear best-point conversion at the
initial head, no p_min/start costs, no head dependence, tunnels as free transfers up to a cap.
"""
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.optimize import linprog
from src.sintef_pipeline import HORIZON, read_time_csv
from src.sintef_research_features import window_sum

MM3 = 0.0036  # Mm3 per (m3/s during one hour)
# SHOP: energy_value_input is local, relative to the reservoir below the FIRST plant under the
# reservoir. This map reproduces the identical water-value groups found in the raw data.
FIRST_PLANT = {'Bandak': 'Hogga', 'Vinjevatn': 'Tokke', 'Byrtevatn': 'Lio', 'Botnedalsvatn': 'Byrte',
               'Vaamarvatn': 'Vinje', 'Totak': 'Vinje', 'Venemo': 'Vinje', 'Hyljelihyl': 'Vinje',
               'Vatjern': 'Haukeli', 'Langeidvatn': 'Haukeli', 'Songavatn': 'Songa', 'Bitdalsvatn': 'Songa',
               'Foersvatn': 'Kjela', 'Bordalsvatn': 'Kjela', 'Langesae': 'Kjela',
               'Kjelavatn': 'Vesle Kjela', 'Staavatn': 'Vesle Kjela'}
PLANT_SOURCE = {'Hogga': 'Bandak', 'Tokke': 'Vinjevatn', 'Lio': 'Byrtevatn', 'Byrte': 'Botnedalsvatn',
                'Vinje': 'Vaamarvatn', 'Haukeli': 'Vatjern', 'Songa': 'Songavatn', 'Kjela': 'Foersvatn',
                'Vesle Kjela': 'Kjelavatn'}
MIN_FLOW_PENALTY = 5000.   # EUR per (m3/s h) of unmet minimum flow; keeps every case feasible
MIN_VOLUME_PENALTY = 1e4   # EUR per (Mm3 h) below a minimum volume (SHOP: soft constraint)
TUNNEL_CAP, TUNNEL_COST = 300., .01  # ponytail: free transfer, no level-dependent tunnel hydraulics


class HydroLP:
    """Static sparse structure is built once; each case changes costs, right-hand sides and bounds."""

    def __init__(self, topology, min_flow_rivers, min_volume_reservoirs):
        model, edges = topology['model'], topology['connections']
        self.model = model
        self.res = list(model['reservoir'])
        self.plants = list(model['plant'])
        self.ridx = {r: i for i, r in enumerate(self.res)}
        down = {}
        for e in edges:
            down.setdefault((e['from'], e['from_type']), []).append((e['to'], e['to_type']))

        def resolve(node, kind, via=False):
            # Follow rivers until a reservoir; r_Vest_Vassdraget is a junction ending in Bandak.
            if kind == 'reservoir':
                return node, via
            following = down.get((node, kind), [])
            if not following:
                return 'SEA', via
            (nxt, nxt_kind), = following
            return resolve(nxt, nxt_kind, via or node == 'r_Vest_Vassdraget')

        arcs = []
        generators = model['generator']
        for p in self.plants:
            units = [g for g in generators if g.rsplit('_G', 1)[0] == p]
            qmax = float(sum(max(max(c['x']) for c in generators[g]['turb_eff_curves']) for g in units))
            outs = down.get((p, 'plant'), [])
            dst, via = resolve(*outs[0]) if outs else ('SEA', False)
            sources = ('Songavatn', 'Bitdalsvatn') if p == 'Songa' else (PLANT_SOURCE[p],)
            for s in sources:  # Songa draws from two tunnel-connected reservoirs
                arcs.append(dict(name=f'plant:{p}:{s}', src=s, dst=dst, cost=0., ub=qmax, plant=p, via=via,
                                 units=len(units)))
        for name, cfg in model['river'].items():
            ups = [e['from'] for e in edges if e['to'] == name and e['to_type'] == 'river' and e['from_type'] == 'reservoir']
            if not ups:
                continue
            (src,) = ups
            dst, via = resolve(name, 'river')
            if 'flow_cost_curve' in cfg:  # piecewise-constant marginal cost, NOK|EUR per (m3/s h)
                x, y = cfg['flow_cost_curve']['x'], cfg['flow_cost_curve']['y']
                for k in range(len(x) - 1):
                    arcs.append(dict(name=f'river:{name}:{k}', src=src, dst=dst, cost=float(y[k]),
                                     ub=float(x[k + 1] - x[k]), plant=None, via=via))
            else:
                arcs.append(dict(name=f'river:{name}', src=src, dst=dst, cost=float(cfg.get('flow_cost_const', 0.)),
                                 ub=float(cfg.get('max_flow_const', np.inf)), plant=None, via=via))
        for name in model['tunnel']:
            a, b = name.split('_', 1)
            if b in model['plant']:
                continue
            for s, d in ((a, b), (b, a)):
                arcs.append(dict(name=f'tunnel:{s}>{d}', src=s, dst=d, cost=TUNNEL_COST, ub=TUNNEL_CAP,
                                 plant=None, via=False))
        self.arcs = arcs
        self.below = {a['plant']: a['dst'] for a in arcs if a['plant']}
        self.plant_arc = {p: [k for k, a in enumerate(arcs) if a['plant'] == p] for p in self.plants}
        self.min_flow_rivers = list(min_flow_rivers)
        self.min_volume_reservoirs = [r for r in min_volume_reservoirs if r in self.ridx]
        T, R, A = HORIZON, len(self.res), len(arcs)
        F, M = len(self.min_flow_rivers), len(self.min_volume_reservoirs)
        self.nf, self.nv = A * T, R * T
        self.nvar = self.nf + self.nv + F * T + M * T
        t = np.arange(T)
        # Balance: V[t+1] - V[t] + MM3*(outflow - inflow) = MM3*natural_inflow (+ V0 at t=0).
        rows = [np.repeat(np.arange(R) * T, T) + np.tile(t, R)]
        cols = [self.nf + rows[0]]
        vals = [np.ones(R * T)]
        prev = np.repeat(np.arange(R) * T, T - 1) + np.tile(t[1:], R)
        rows.append(prev); cols.append(self.nf + prev - 1); vals.append(-np.ones(len(prev)))
        for k, arc in enumerate(arcs):
            for node, sign in ((arc['src'], 1.), (arc['dst'], -1.)):
                if node in self.ridx:
                    rows.append(self.ridx[node] * T + t); cols.append(k * T + t); vals.append(np.full(T, sign * MM3))
        self.A_eq = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(R * T, self.nvar))
        rows, cols, vals, row = [], [], [], 0
        for k, river in enumerate(self.min_flow_rivers):
            members = [a for a, arc in enumerate(arcs)
                       if (arc['via'] if river == 'r_Vest_Vassdraget' else arc['name'] == f'river:{river}')]
            if not members:
                raise ValueError(f'No arcs for minimum-flow river {river}')
            for a in members:
                rows.append(row + t); cols.append(a * T + t); vals.append(-np.ones(T))
            rows.append(row + t); cols.append(self.nf + self.nv + k * T + t); vals.append(-np.ones(T))
            row += T
        for k, name in enumerate(self.min_volume_reservoirs):
            rows.append(row + t); cols.append(self.nf + self.ridx[name] * T + t); vals.append(-np.ones(T))
            rows.append(row + t); cols.append(self.nf + self.nv + F * T + k * T + t); vals.append(-np.ones(T))
            row += T
        self.capacity_rows = []
        for p, members in self.plant_arc.items():
            if len(members) > 1:  # shared turbine capacity for multi-intake plants
                for a in members:
                    rows.append(row + t); cols.append(a * T + t); vals.append(np.ones(T))
                self.capacity_rows.append((row, arcs[members[0]]['ub']))
                row += T
        self.A_ub = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(row, self.nvar))
        self.lb = np.zeros(self.nvar)
        self.ub_template = np.full(self.nvar, np.inf)
        for k, arc in enumerate(arcs):
            self.ub_template[k * T:(k + 1) * T] = arc['ub']
        self.cost_template = np.zeros(self.nvar)
        for k, arc in enumerate(arcs):
            self.cost_template[k * T:(k + 1) * T] = arc['cost']
        self.cost_template[self.nf + self.nv:self.nf + self.nv + F * T] = MIN_FLOW_PENALTY
        self.cost_template[self.nf + self.nv + F * T:] = MIN_VOLUME_PENALTY

    def conversion(self, plant, level):
        """MW per m3/s at one unit's best point and the given gross level (SHOP converts WV here)."""
        cfg = self.model['plant'][plant]
        gens = [g for g in self.model['generator'] if g.rsplit('_G', 1)[0] == plant]
        head = level - float(cfg['outlet_line'])
        curve = min(self.model['generator'][gens[0]]['turb_eff_curves'], key=lambda c: abs(c['ref'] - head))
        k = int(np.argmax(curve['y']))
        q, eta = float(curve['x'][k]), float(curve['y'][k]) / 100
        penstock = max([float(v) for v in cfg.get('penstock_loss') or [0.]])
        loss = float(cfg['main_loss'][0]) * (q * len(gens)) ** 2 + penstock * q ** 2
        return 9.81e-3 * max(head - loss, 1.) * eta * .98

    def solve(self, price, inflow, v0, wv_end, min_flow, min_volume):
        """price (T,); inflow: node -> (T,); v0/wv_end: reservoir -> float;
        min_flow: river -> (T,); min_volume: reservoir -> (T,). Returns plant x 6 x T array."""
        T, R = HORIZON, len(self.res)
        level = {r: np.interp(v0[r], self.model['reservoir'][r]['vol_head']['x'],
                              self.model['reservoir'][r]['vol_head']['y']) for r in self.res}
        e = {p: self.conversion(p, level[PLANT_SOURCE[p]]) for p in self.plants}
        G = {}

        def global_value(r):  # EUR per Mm3: local value x conversion + value below the plant
            if r == 'SEA':
                return 0.
            if r not in G:
                p = FIRST_PLANT[r]
                G[r] = wv_end[r] * e[p] / MM3 + global_value(self.below[p])
            return G[r]
        c = self.cost_template.copy()
        for k, arc in enumerate(self.arcs):
            if arc['plant']:
                c[k * T:(k + 1) * T] -= price * e[arc['plant']]
        ub = self.ub_template.copy()
        for i, r in enumerate(self.res):
            c[self.nf + i * T + T - 1] -= global_value(r)
            ub[self.nf + i * T:self.nf + (i + 1) * T] = max(float(self.model['reservoir'][r]['max_vol']), v0[r])
        b_eq = np.zeros(R * T)
        for i, r in enumerate(self.res):
            b_eq[i * T:(i + 1) * T] = MM3 * inflow[r]
            b_eq[i * T] += v0[r]
        b_eq[self.ridx['Bandak'] * T:(self.ridx['Bandak'] + 1) * T] += MM3 * inflow['r_Vest_Vassdraget']
        b_ub = []
        for river in self.min_flow_rivers:
            natural = inflow['r_Vest_Vassdraget'] if river == 'r_Vest_Vassdraget' else 0.
            b_ub.append(-(min_flow[river] - natural))
        for r in self.min_volume_reservoirs:
            b_ub.append(-min_volume[r])
        for _, cap in self.capacity_rows:
            b_ub.append(np.full(T, cap))
        out = linprog(c, A_ub=self.A_ub, b_ub=np.concatenate(b_ub), A_eq=self.A_eq, b_eq=b_eq,
                      bounds=np.column_stack([self.lb, ub]), method='highs-ds')
        if out.status != 0:
            raise RuntimeError(f'LP failed: {out.message}')
        x = out.x
        lam = -out.eqlin.marginals.reshape(R, T)  # value of one extra Mm3 at (reservoir, hour)
        reduced = out.lower.marginals + out.upper.marginals  # objective change per unit flow
        result = np.empty((len(self.plants), 6, T))
        for j, p in enumerate(self.plants):
            members = self.plant_arc[p]
            q = sum(x[a * T:(a + 1) * T] for a in members) / self.arcs[members[0]]['ub']
            s, d = self.ridx[PLANT_SOURCE[p]], self.below[p]
            lam_d = lam[self.ridx[d]] if d in self.ridx else np.zeros(T)
            vol = lambda i: np.concatenate([[v0[self.res[i]]], x[self.nf + i * T:self.nf + (i + 1) * T - 1]])
            result[j] = [q, MM3 * (lam[s] - lam_d) / e[p], -reduced[members[0] * T:(members[0] + 1) * T] / e[p],
                         vol(s), vol(self.ridx[d]) if d in self.ridx else np.zeros(T), np.full(T, e[p])]
        return result


def lp_inputs(root, starts):
    """Raw per-case inputs, exactly the supplied case window; no later observed storage."""
    with open(root / 'data/extended/Tokke_Vinje_topology.yaml') as stream:
        topology = yaml.safe_load(stream)
    starts = pd.DatetimeIndex(starts)
    n = len(starts)
    times = starts.repeat(HORIZON) + pd.to_timedelta(np.tile(np.arange(HORIZON), n), unit='h')

    def hourly(path):
        frame = read_time_csv(path).set_index('date').select_dtypes(include=np.number).reindex(times)
        if frame.isna().any().any():
            raise ValueError(f'Missing hourly inputs in {path.name}')
        return {c: frame[c].to_numpy(np.float64).reshape(n, HORIZON) for c in frame}

    def daily(path, when):
        frame = read_time_csv(path).set_index('date').select_dtypes(include=np.number).reindex(when)
        if frame.isna().any().any():
            raise ValueError(f'Missing daily inputs in {path.name}')
        return frame
    price = read_time_csv(root / 'data/kernel/Historical_day_ahead_price_2015_2025.csv').set_index('date')
    price = price['Day-ahead price (EUR/MWh)'].reindex(times).to_numpy(np.float64).reshape(n, HORIZON)
    return dict(topology=topology, price=price,
                inflow=hourly(root / 'data/kernel/Historical_inflow_1958_2025.csv'),
                min_flow=hourly(root / 'data/extended/Constraint_min_flow.csv'),
                min_volume=hourly(root / 'data/extended/Constraint_min_volume.csv'),
                v0=daily(root / 'data/kernel/Historical_volume_2015_2024.csv', starts),
                wv_end=daily(root / 'data/kernel/Synthetic_water_value_2015_2024.csv', starts + pd.Timedelta(days=7)))


def solve_all_cases(root, starts, threads=1):
    """Deterministic dual simplex per case; threads only change speed, not results.
    Returns (cases x plants x 6 x T cube, plant names, seconds)."""
    from concurrent.futures import ThreadPoolExecutor
    inputs = lp_inputs(root, starts)
    lp = HydroLP(inputs['topology'], inputs['min_flow'], inputs['min_volume'])
    n = len(inputs['price'])

    def one(k):
        return lp.solve(inputs['price'][k], {c: v[k] for c, v in inputs['inflow'].items()},
                        inputs['v0'].iloc[k].to_dict(), inputs['wv_end'].iloc[k].to_dict(),
                        {c: v[k] for c, v in inputs['min_flow'].items()},
                        {c: v[k] for c, v in inputs['min_volume'].items()})
    started = time.perf_counter()
    with ThreadPoolExecutor(max(1, int(threads))) as pool:
        cube = np.stack(list(pool.map(one, range(n)))).astype(np.float32)
    if not np.isfinite(cube).all():
        raise ValueError('Non-finite LP output')
    return cube, lp.plants, time.perf_counter() - started


def lp_plant_features(cube, plants, price, wv_source, units_per_plant):
    """Per-plant model inputs from the LP solution. wv_source: plant -> (cases,) terminal local value."""
    n = len(price)
    scale_price = price.std(axis=1)[:, None]
    frames = {}
    for j, p in enumerate(plants):
        q, threshold, rc_margin, vol_src, vol_dst, _ = (cube[:, j, k].astype(np.float64) for k in range(6))
        wv = np.asarray(wv_source[p], np.float64)[:, None]
        scale = scale_price + np.abs(wv) + 1.
        out = {
            'lp_discharge_fraction': q,
            'lp_week_fraction': np.broadcast_to(q.mean(axis=1, keepdims=True), q.shape),
            'lp_day_fraction': np.repeat(q.reshape(n, 7, 24).mean(axis=2), 24, axis=1),
            'lp_next_24h_fraction': window_sum(q, 24) / np.minimum(24, np.arange(HORIZON, 0, -1)),
            'lp_past_24h_fraction': window_sum(q, 24, False) / np.minimum(24, np.arange(1, HORIZON + 1)),
            'lp_reduced_cost_margin': rc_margin,
            'lp_reduced_cost_margin_normalized': rc_margin / scale,
            'lp_dual_margin': price - threshold,
            'lp_dual_margin_normalized': (price - threshold) / scale,
            'lp_threshold_shift_normalized': (threshold - wv) / scale,
            'lp_source_volume_mm3': vol_src,
            'lp_downstream_volume_mm3': vol_dst,
        }
        if units_per_plant[p] > 1:
            out['lp_units_equivalent'] = q * units_per_plant[p]
        frame = pd.DataFrame({k: np.asarray(v, np.float32).ravel() for k, v in out.items()})
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f'Non-finite LP feature for {p}')
        frames[p] = frame
    return frames
