"""Synthetic mathematical checks only: no real-data model fitting."""
import ast
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.sintef_structure_probe import (
    chain_marginals, smooth_local, fit_plant_priors, apply_plant_priors,
    timing_metrics, candidates, select_on_development,
)


def test_forward_backward_matches_exhaustive_enumeration():
    p=np.array([.2,.8,.35,.7])
    states=np.array(list(itertools.product([0,1],repeat=len(p))))
    weight=np.prod(np.where(states,p,1-p),axis=1)*np.exp(-.4*np.abs(np.diff(states,axis=1)).sum(1))
    expected=weight@states/weight.sum()
    actual=chain_marginals(p[None,None,:],.4).ravel()
    np.testing.assert_allclose(actual,expected,atol=1e-7)


def test_zero_strength_identity_and_case_boundaries():
    p=np.random.default_rng(8).uniform(.1,.9,(3,2,12)).astype(np.float32)
    np.testing.assert_array_equal(chain_marginals(p,0),p)
    np.testing.assert_array_equal(smooth_local(p,0),p)
    for fn in (chain_marginals,smooth_local):
        together=fn(p)
        separate=np.concatenate([fn(p[i:i+1]) for i in range(len(p))])
        np.testing.assert_allclose(together,separate,atol=1e-7)


def test_chain_one_hour_and_extremes():
    p=np.array([[[.2]],[[.8]]],dtype=np.float32)
    np.testing.assert_allclose(chain_marginals(p),p,atol=1e-7)
    q=chain_marginals(np.array([[[0.,0.,1.,1.]]]))
    assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()


def test_joint_prior_independence_and_single_units_unchanged():
    units=['A_G1','A_G2','B_G1']
    y=np.array([[[0,0,1,1],[0,1,0,1],[1,1,0,0]]])
    prior=fit_plant_priors(y,units,prior_mix=0)
    p=np.random.default_rng(2).uniform(.1,.9,y.shape).astype(np.float32)
    for kind in ['count','state']:
        np.testing.assert_allclose(apply_plant_priors(p,prior,kind),p,atol=1e-7)
        np.testing.assert_array_equal(apply_plant_priors(p,prior,kind)[:,2],p[:,2])


def test_prior_soft_probability_and_unit_axis_alignment():
    units=['A_G1','B_G1','A_G2']
    y=np.array([[[0,0,1,1],[1,0,1,0],[0,0,1,1]]])
    prior=fit_plant_priors(y,units)
    p=np.array([[[.2,.2,.8,.8],[.3,.4,.5,.6],[.2,.2,.8,.8]]],dtype=np.float32)
    for kind in ['count','state']:
        q=apply_plant_priors(p,prior,kind)
        assert q.shape==p.shape and np.isfinite(q).all()
        assert ((q>0)&(q<1)).all()
        np.testing.assert_array_equal(q[:,1],p[:,1])


def test_no_validation_label_access_and_purge():
    rng=np.random.default_rng(9)
    train=dict(starts=pd.date_range('2021-12-22',periods=10,tz='UTC'),
        truth=rng.integers(0,2,(10,2,168)))
    p=rng.uniform(.05,.95,(3,2,168)).astype(np.float32)
    valid=dict(starts=pd.date_range('2022-01-01',periods=3,tz='UTC'),base=p,v5=p*.9)
    cfg=dict(prior_mix=.25,prior_strength=.25,smooth_weight=.15,
             switch_penalty=.4,chain_blend=.25,v5_weight=.1)
    a,prior,split=candidates(train,valid,['A_G1','A_G2'],cfg)
    assert split==dict(training_weeks=4,purged_weeks=6)
    train['truth'][4:]=1-train['truth'][4:]
    b,_,_=candidates(train,valid,['A_G1','A_G2'],cfg)
    assert len(a)==6
    for name in a:np.testing.assert_allclose(a[name],b[name])


def test_within_week_auc_matches_brute_force_with_ties():
    y=np.array([[[1,0,1,0]],[[1,1,1,1]]])
    p=np.array([[[.8,.2,.5,.5]],[[.1,.2,.3,.4]]])
    assert timing_metrics(y,p)['within_week_auc']==.875


def test_selection_uses_development_and_fallback():
    t=pd.DataFrame([dict(group='all',model='original_lp',micro_auc=.99),
                    dict(group='all',model='a',micro_auc=.990005),
                    dict(group='all',model='b',micro_auc=.98)])
    assert select_on_development(t)['selected']=='original_lp'
    t.loc[1,'micro_auc']=.991
    assert select_on_development(t)['selected']=='a'


def test_notebook_compiles_all_writes_in_new_directory():
    root=Path(__file__).resolve().parents[1]
    nb=json.loads((root/'notebooks/21_structure_and_timing_probe.ipynb').read_text())
    for c in nb['cells']:
        if c['cell_type']!='code':continue
        s=''.join(c['source']);compile(s,'notebook21','exec')
        for node in ast.walk(ast.parse(s)):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                if node.func.attr in ('to_csv','savez_compressed'):
                    assert ast.unparse(node.args[0]).startswith('OUT /')
                if node.func.attr=='write_text':
                    assert ast.unparse(node.func.value).startswith('OUT /')
    s='\n'.join(''.join(c['source']) for c in nb['cells'])
    assert s.index('selection_before_2022.json') < s.index('audit_predictions, audit_priors')
    assert 'exist_ok=False' in s and 'ORIGINAL_HASHES' in s
