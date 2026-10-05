"""Small, explicit structural probes of saved predictions. No file writes or boosting.

The chain penalty is dimensionless regularization, NOT SHOP's monetary start cost.
Plant priors are soft empirical corrections, never hard feasibility constraints.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp, softmax
from scipy.stats import rankdata
from src.sintef_weekly_probe import purged_training_mask, score_predictions


def checked(p):
    p = np.asarray(p)
    if p.ndim != 3 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite probability cube: cases, units, hours.')
    return p


def smooth_local(p, weight=.15):
    """3-hour logit average, truncated (not repeated) at the case boundaries."""
    p = checked(p)
    if not 0 <= weight <= 1:
        raise ValueError('Invalid smoothing weight.')
    if weight == 0:
        return p.copy()
    z = logit(np.clip(p.astype(float), 1e-7, 1-1e-7))
    sums = z.copy()
    count = np.ones(z.shape[-1])
    sums[..., 1:] += z[..., :-1]; count[1:] += 1
    sums[..., :-1] += z[..., 1:]; count[:-1] += 1
    return expit((1-weight)*z + weight*sums/count).astype(np.float32)


def chain_marginals(p, switch_penalty=.4):
    """Exact marginals of product Bernoulli scores times exp(-penalty * switches).

    Free boundary states: no assumed ON/OFF before/after the 168-hour case.
    All dynamic programming is inside each case/unit. No Viterbi/hard labels.
    """
    p = checked(p)
    if switch_penalty < 0 or not np.isfinite(switch_penalty):
        raise ValueError('Invalid penalty.')
    if switch_penalty == 0:
        return p.copy()
    shape = p.shape
    z = np.clip(p.astype(float).reshape(-1, shape[-1]), 1e-7, 1-1e-7)
    emission = np.stack([np.log1p(-z), np.log(z)], axis=-1)
    transition = np.array([[0., -switch_penalty], [-switch_penalty, 0.]])
    forward = np.empty_like(emission)
    forward[:, 0] = emission[:, 0]
    forward[:, 0] -= logsumexp(forward[:, 0], axis=1)[:, None]
    for t in range(1, shape[-1]):
        forward[:, t] = emission[:, t] + logsumexp(forward[:, t-1, :, None] + transition, axis=1)
        forward[:, t] -= logsumexp(forward[:, t], axis=1)[:, None]
    backward = np.zeros((len(z), 2))
    result = np.empty_like(z)
    for t in range(shape[-1]-1, -1, -1):
        result[:, t] = softmax(forward[:, t]+backward, axis=1)[:, 1]
        if t:
            backward = logsumexp(transition[None, :, :] + emission[:, t, None, :] + backward[:, None, :], axis=2)
            backward -= logsumexp(backward, axis=1)[:, None]
    return result.reshape(shape).astype(np.float32)


def fit_plant_priors(truth, units, prior_mix=.25):
    """Empirical joint-state and count priors, shrunk toward independent marginals.

    Only training labels may enter. Rolling cases are dependent: counts are NOT
    treated as independent evidence for significance. Max 4 units in this task.
    """
    if not 0 <= prior_mix <= 1:
        raise ValueError('Invalid prior_mix.')
    y = np.asarray(truth)
    if y.ndim != 3 or y.shape[1] != len(units) or not np.isin(y, [0, 1]).all() or not len(y):
        raise ValueError('Invalid training labels.')
    priors = {}
    for plant in dict.fromkeys(u.rsplit('_G', 1)[0] for u in units):
        ids = [i for i, u in enumerate(units) if u.rsplit('_G', 1)[0] == plant]
        k = len(ids)
        if k == 1:
            continue
        if k > 4:
            raise ValueError('This bounded probe only supports <=4 units/plant.')
        states = ((np.arange(2**k)[:, None] >> np.arange(k)) & 1).astype(float)
        labels = y[:, ids].transpose(0, 2, 1).reshape(-1, k)
        marginal = (labels.sum(0)+1)/(len(labels)+2)
        independent = np.prod(np.where(states, marginal, 1-marginal), axis=1)
        encoded = (labels * (2**np.arange(k))).sum(1).astype(int)
        joint = (np.bincount(encoded, minlength=2**k)+1)/(len(labels)+2**k)
        mixed = (1-prior_mix)*independent + prior_mix*joint
        count = states.sum(1).astype(int)
        independent_count = np.bincount(count, weights=independent, minlength=k+1)
        mixed_count = np.bincount(count, weights=mixed, minlength=k+1)
        priors[plant] = dict(ids=ids, states=states,
            state_effect=np.clip(np.log(mixed/independent), -3, 3),
            count_effect=np.clip(np.log(mixed_count/independent_count)[count], -3, 3))
    return priors


def apply_plant_priors(p, priors, kind='count', strength=.25):
    """Marginalize all soft joint states, retaining unit identity; never choose a hard state."""
    p = checked(p)
    if kind not in ('count', 'state') or strength < 0 or not np.isfinite(strength):
        raise ValueError('Invalid prior configuration.')
    if strength == 0:
        return p.copy()
    out = p.copy()
    for prior in priors.values():
        ids, states = prior['ids'], prior['states']
        v = np.clip(p[:, ids].transpose(0, 2, 1).reshape(-1, len(ids)), 1e-7, 1-1e-7)
        # OFF log-probability is a common normalizer and can be omitted.
        scores = logit(v.astype(float)) @ states.T + strength*prior[f'{kind}_effect']
        marginal = softmax(scores, axis=1) @ states
        out[:, ids] = marginal.reshape(p.shape[0], p.shape[-1], len(ids)).transpose(0, 2, 1)
    return out.astype(np.float32)


def candidates(train, valid, units, config):
    """Uses only purged training truth; validation truth is neither required nor read."""
    keep = purged_training_mask(train['starts'], valid['starts'])
    prior = fit_plant_priors(train['truth'][keep], units, config['prior_mix'])
    p = checked(valid['base'])
    old = checked(valid['v5'])
    if old.shape != p.shape:
        raise ValueError('Base/control predictions differ in shape.')
    w, hedge = config['chain_blend'], config['v5_weight']
    if not 0 <= w <= 1 or not 0 <= hedge <= 1:
        raise ValueError('Invalid blend weight.')
    predictions = dict(original_lp=p.copy(),
        local_smoothing=smooth_local(p, config['smooth_weight']),
        soft_switch_chain=((1-w)*p+w*chain_marginals(p, config['switch_penalty'])).astype(np.float32),
        plant_count_prior=apply_plant_priors(p, prior, 'count', config['prior_strength']),
        plant_state_prior=apply_plant_priors(p, prior, 'state', config['prior_strength']),
        conservative_v5_blend=((1-hedge)*p+hedge*old).astype(np.float32))
    return predictions, prior, dict(training_weeks=int(keep.sum()), purged_weeks=int((~keep).sum()))


def timing_metrics(y, p):
    """Within-week AUC weighted by pos*neg; actual transitions used only for diagnostics."""
    positive = y.sum(axis=-1)
    pairs = positive*(y.shape[-1]-positive)
    ranks = rankdata(p, axis=-1, method='average')
    concordance = (ranks*y).sum(axis=-1)-positive*(positive+1)/2
    transitions = y[..., 1:] != y[..., :-1]
    near = np.zeros_like(y, dtype=bool)
    near[..., 1:] |= transitions
    near[..., :-1] |= transitions
    return dict(within_week_auc=float(concordance.sum()/pairs.sum()) if pairs.sum() else np.nan,
        transition_brier=float(((p-y)[near]**2).mean()) if near.any() else np.nan,
        soft_temporal_variation=float(np.abs(np.diff(p,axis=-1)).mean()))


def evaluate_candidates(bundle, predictions, units):
    table = score_predictions(bundle['truth'], predictions, units, bundle['starts'])
    timing = pd.DataFrame([dict(model=name, **timing_metrics(bundle['truth'], p)) for name,p in predictions.items()])
    return table, timing


def select_on_development(table, minimum_gain=.00001):
    """One global choice using development only; never per-unit cherry-picking."""
    overall = table[table.group == 'all'].set_index('model')
    baseline = float(overall.loc['original_lp', 'micro_auc'])
    # Sorting names also makes ties deterministic. Tiny gains fall back to original.
    winners = sorted((float(r.micro_auc)-baseline, str(name)) for name,r in overall.iterrows()
                     if name != 'original_lp')
    gain, name = winners[-1] if winners else (0., 'original_lp')
    return dict(selected=name if gain > minimum_gain else 'original_lp',
                development_gain=gain if gain > minimum_gain else 0., minimum_gain=minimum_gain,
                note='Exploratory point estimates, no statistical confidence claim.')
