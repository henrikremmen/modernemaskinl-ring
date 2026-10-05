"""Pure/mocked tests: never train a real estimator or touch original CSVs."""
import ast
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.sintef_weekly_probe import (
    level_shift, purged_training_mask, run_statistics, weekly_features,
    run_probe, score_predictions, error_diagnostics,
)


def toy_bundle(year=2021, n=10):
    rng = np.random.default_rng(8)
    base = rng.uniform(.1,.9,(n,1,168)).astype(np.float32)
    lp = np.zeros((n,1,6,168),np.float32)
    lp[:,:,0] = .5
    lp[:,:,1] = 20
    lp[:,:,3:5] = 5
    lp[:,:,5] = .2
    return dict(base=base, v5=base*.9, lp=lp, price=np.full((n,168),30.),
                truth=(base>.5).astype(np.int8),
                starts=pd.date_range(f'{year}-01-01',periods=n,tz='UTC'))


def toy_features(bundle):
    topology={'model':{'generator':{'A_G1':dict(p_min=1.,p_max=10.,
        startcost_const=10.,stopcost_const=10.,turb_eff_curves=[{'x':[0.,50.]}])}}}
    return weekly_features(bundle,['A_G1'],['A'],topology)


def test_purge_excludes_six_overlapping_cases():
    starts=pd.date_range('2021-01-01','2021-12-31',tz='UTC')
    keep=purged_training_mask(starts,pd.date_range('2022-01-01',periods=5,tz='UTC'))
    assert keep.sum()==359
    assert list(starts[~keep].day)==[26,27,28,29,30,31]


def test_level_shift_hits_reachable_mean_and_keeps_order():
    p=np.array([[.1,.2,.5,.8],[.2,.3,.6,.9]])
    corrected, shifts=level_shift(p,p.mean(1)+.04)
    np.testing.assert_allclose(corrected.mean(1),p.mean(1)+.04,atol=1e-6)
    assert (np.diff(corrected,axis=1)>0).all()
    assert (shifts>0).all()
    bounded,shift=level_shift(p,np.ones(2),max_logit_shift=.2)
    assert np.max(np.abs(shift))<=.2
    assert np.isfinite(bounded).all()


def test_noop_and_extreme_predictions():
    p=np.array([[0.,.3,1.]],dtype=np.float32)
    q,shift=level_shift(p,p.mean(1,dtype=float))
    np.testing.assert_array_equal(q,p)
    np.testing.assert_array_equal(shift,[0.])
    q,_=level_shift(np.zeros((2,168)),np.ones(2)*.1)
    assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    with pytest.raises(ValueError): level_shift(p,np.array([np.nan]))


def test_features_have_no_access_to_labels():
    b=toy_bundle()
    with_truth=toy_features(b)
    del b['truth']
    without_truth=toy_features(b)
    for arm in with_truth:
        pd.testing.assert_frame_equal(with_truth[arm]['A_G1'],without_truth[arm]['A_G1'])
    assert len(with_truth['enhanced']['A_G1'].columns)>len(with_truth['control']['A_G1'].columns)


def test_run_statistics_do_not_cross_week_boundaries():
    m=np.zeros((2,168));m[0,-3:]=2;m[1,:2]=2
    s=run_statistics(m,np.ones_like(m),switching_cost=5)
    np.testing.assert_array_equal(s[:,1],[3,2])
    np.testing.assert_allclose(s[:,3],[3/168,0])


def test_mocked_training_purges_and_never_reads_validation_truth():
    tr,va=toy_bundle(2021),toy_bundle(2022)
    tr['starts']=pd.date_range('2021-12-22',periods=10,tz='UTC')
    tf,vf=toy_features(tr),toy_features(va)
    del va['truth']
    class FakeModel:
        def fit(self,x,y):
            assert len(x)==4 and len(y)==4
            self.coef_=np.zeros(x.shape[1]);return self
        def predict(self,x):return np.zeros(len(x))
        def __getitem__(self,key):return self
    with patch('sklearn.pipeline.make_pipeline',side_effect=lambda *a:FakeModel()):
        out,info,coefs=run_probe(tr,va,tf,vf,['A_G1'])
    assert len(info)==2 and (info.purged==6).all()
    for p in out.values():np.testing.assert_allclose(p,va['base'],atol=1e-6)
    assert len(coefs)>0


def test_metrics_and_error_groups():
    b=toy_bundle()
    scores=score_predictions(b['truth'],{'base':b['base']},['A_G1'],b['starts'])
    assert len(scores)==3
    assert scores.query("group=='all'").micro_auc.iloc[0]==1
    d=error_diagnostics(b,['A_G1'])
    assert d.query("group in ['near_true_transition','away_from_transition']").hours.sum()==10*168


def test_notebook_compiles_and_only_exports_to_out():
    root=Path(__file__).resolve().parents[1]
    nb=json.loads((root/'notebooks/20_weekly_correction_probe.ipynb').read_text())
    for c in nb['cells']:
        if c['cell_type']!='code':continue
        assert c['execution_count'] is None and c['outputs']==[]
        source=''.join(c['source'])
        compile(source,'notebook20','exec')
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                if node.func.attr in ('to_csv','savez_compressed'):
                    assert ast.unparse(node.args[0]).startswith('OUT /')
                if node.func.attr=='write_text':
                    assert ast.unparse(node.func.value).startswith('OUT /')
    full='\n'.join(''.join(c['source']) for c in nb['cells'])
    assert 'exist_ok=False' in full and 'ORIGINAL_CSV_HASHES' in full
    assert 'RUN_SMALL_MODELS = True' in full
