"""Feature, resume and standalone export checks; only tiny real API fits."""
import ast
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from src.sintef_path_features import contiguous_competition, weekly_shape
from src.sintef_path_runner import PathRunner, select_pair, staged_predict, scaled_rounds, final_paths
from src.sintef_challenger_runner import fit_member


def test_contiguous_competition_bruteforce():
    p=np.array([[1,4,3,2,5,1],[3,3,3,3,3,3]],dtype=float)
    result=contiguous_competition(p)
    for row in range(len(p)):
        for t in range(p.shape[1]):
            for label,direction in [('next',1),('past',-1)]:
                count,premium=0,0.
                k=t+direction
                while 0<=k<p.shape[1] and p[row,k]>p[row,t]:
                    count+=1;premium+=p[row,k]-p[row,t];k+=direction
                assert result[f'path_consecutive_better_{label}'][row,t]==count
                assert result[f'path_contiguous_premium_{label}'][row,t]==premium


def test_weekly_shape_no_cross_case_and_known_day_means():
    p=np.stack([np.arange(168),np.full(168,3.)]).astype(float)
    result=weekly_shape(p)
    changed=weekly_shape(np.stack([p[0],np.full(168,1e6)]))
    for name in result:
        np.testing.assert_array_equal(result[name][0],changed[name][0])
        assert np.isfinite(result[name]).all()
    np.testing.assert_allclose(result['path_day0_mean_gap'][0],(11.5-p[0])/(p[0].std()+1),atol=1e-7)
    np.testing.assert_allclose(result['path_price_dct_1'][1],0,atol=1e-7)
    assert len(result)==30


def make_runner(tmp_path,monkeypatch,use_checkpoints=True):
    import src.sintef_path_runner as module
    monkeypatch.setattr(module,'raw_digest',lambda root:'test_raw')
    units=['A_G1','A_G2']
    target=np.zeros((5,2,168),np.int8);target[:,0,::2]=1
    data=SimpleNamespace(root=tmp_path,units=units,n_train_cases=5,n_all_cases=7,
        target_cube=target,X_all=pd.DataFrame({'price':np.arange(7*168,dtype=np.float32)/100}),
        feature_sets={'full_engineered':['price']},n_test_cases=2,
        test_cases=pd.DataFrame({'Run No':[5,6]}),sample_submission=pd.DataFrame(columns=['Run No']+[f'result_committed_{u}_t{h}' for u in units for h in range(168)]))
    data.sample_submission=pd.DataFrame(np.zeros((2,337)),columns=data.sample_submission.columns)
    data.sample_submission['Run No']=[5,6]
    extra=pd.DataFrame({'global':np.zeros(7*168,dtype=np.float32)})
    local={'A':pd.DataFrame({'local':np.zeros(7*168,dtype=np.float32)})}
    competition={'A':pd.DataFrame({'competition':np.ones(7*168,dtype=np.float32)})}
    paths={'A':pd.DataFrame({'path':np.ones(7*168,dtype=np.float32)*2})}
    specs={f:dict(backend=b,features='competition',parameters={}) for f,b in [('xgb','xgboost'),('lgb','lightgbm')]}
    rounds={f:dict(zip(units,[2,4])) for f in specs}
    return PathRunner(data,extra,local,competition,paths,specs,rounds,1,'source','v4_source',use_checkpoints)


class FakeModel:
    def __init__(self,width):
        self.feature_importances_=np.ones(width)
    def predict_proba(self,X,iteration_range):
        p=expit(X[:,0]/10+iteration_range[1]/100)
        return np.column_stack([1-p,p])
    def predict(self,X,num_iteration,num_threads):
        return self.predict_proba(X,(0,num_iteration))[:,1]
    def feature_importance(self,importance_type):
        return self.feature_importances_


def test_resume_source_invalidation_and_constant_unit(tmp_path,monkeypatch):
    import src.sintef_path_runner as module
    calls=[]
    def fake_fit(spec,X,y,rounds,threads):
        calls.append(rounds);return FakeModel(X.shape[1]),rounds
    monkeypatch.setattr(module,'fit_member',fake_fit)
    runner=make_runner(tmp_path,monkeypatch)
    first=runner.run('xgb','path',[0,1],[2,3],'test')
    assert calls==[4]  # Second unit is constant and needs no estimator.
    second=runner.run('xgb','path',[0,1],[2,3],'test')
    assert calls==[4]
    for k in first:
        np.testing.assert_array_equal(first[k],second[k])
        np.testing.assert_array_equal(first[k][:,1],0)
    runner.source_digest='changed'
    runner.run('xgb','path',[0,1],[2,3],'test');assert calls==[4,4]
    runner.use_checkpoints=False
    runner.run('xgb','path',[0,1],[2,3],'test');assert calls==[4,4,4]
    with pytest.raises(ValueError,match='overlapping'):
        runner.run('xgb','path',[0,1],[1,2],'bad')


def test_selection_ties_favour_existing_simpler_pair():
    truth=np.tile([0,1],168).reshape(1,2,168)
    p=.1+.8*truth
    rows=[];predictions={}
    for family in ('xgb','lgb'):
        for variant in ('path','v4'):
            for factor in (2.,1.5,1.):
                rows.append(dict(family=family,variant=variant,factor=factor,micro_auc=1.))
                predictions[(family,variant,f'{factor:g}')]=p
    selection,_=select_pair(rows,predictions,truth)
    assert selection['xgb_weight']==.5
    assert selection['members']=={'xgb':dict(variant='v4',factor=1.),'lgb':dict(variant='v4',factor=1.)}
    assert scaled_rounds(3,1.5)==4
    with pytest.raises(ValueError):scaled_rounds(10,3)


@pytest.mark.parametrize('backend',['xgboost','lightgbm'])
def test_real_tiny_staged_api(backend):
    rng=np.random.default_rng(42)
    X=rng.normal(size=(64,4)).astype(np.float32);y=(X[:,0]+X[:,1]>0).astype(int)
    if backend=='xgboost':
        params=dict(objective='binary:logistic',max_depth=2,min_child_weight=1,learning_rate=.1,tree_method='hist',device='cpu',random_state=42)
    else:
        params=dict(objective='binary',num_leaves=4,min_data_in_leaf=2,learning_rate=.1,verbosity=-1,seed=42,deterministic=True,force_col_wise=True)
    spec=dict(backend=backend,parameters=params)
    longer,_=fit_member(spec,X,y,4,1)
    shorter,_=fit_member(spec,X,y,2,1)
    a=staged_predict(longer,backend,X,2,1);b=staged_predict(shorter,backend,X,2,1)
    np.testing.assert_allclose(a,b,atol=1e-7)


def test_export_is_valid_self_contained_and_cold(tmp_path,monkeypatch):
    project=Path(__file__).resolve().parents[1]
    (tmp_path/'notebooks').mkdir()
    original=project/'notebooks/18_multiscale_boosting.ipynb'
    shutil.copyfile(original,tmp_path/'notebooks'/original.name)
    nb=json.loads(original.read_text())
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    runner=make_runner(tmp_path,monkeypatch)
    selection=dict(identity=runner.identity(),members={f:dict(variant='path',factor=1.) for f in ('xgb','lgb')},xgb_weight=.5)
    namespace=dict(ROOT=tmp_path,runner=runner,selection=selection,json=json,submission_path='candidate.csv')
    export=''.join(next(c['source'] for c in nb['cells'] if c.get('id')=='export'))
    exec(compile(export,'export-cell','exec'),namespace)
    short=json.loads((runner.directory/'Short_notebook_multiscale_v5.ipynb').read_text())
    ids=[c.get('id') for c in short['cells']]
    assert 'development' not in ids and 'audit' not in ids and 'export' not in ids
    code='\n'.join(''.join(c['source']) for c in short['cells'] if c['cell_type']=='code')
    assert "RUN_PROFILE = 'professor'" in code
    assert 'from src.' not in code and 'def develop_paths(' not in code
    for c in short['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    # Execute the actual exported definitions, not the imported source implementation.
    ns={}
    for c in short['cells']:
        if c.get('id') in ('resources','pipeline','physics','fit_helpers','competition','paths','runner'):
            exec(compile(''.join(c['source']),c['id'],'exec'),ns)
    assert ns['THREADS']<=4 and ns['USE_CHECKPOINTS'] is False
    assert 'final_paths' in ns
    # Run the complete exported final path on toy data using a fake estimator.
    ns['raw_digest']=lambda root:'test_raw'
    ns['fit_member']=lambda spec,X,y,rounds,threads:(FakeModel(X.shape[1]),rounds)
    embedded=ns['PathRunner'](runner.data,pd.DataFrame({'global':np.zeros(7*168)}),runner.local,
        runner.competition,runner.paths,runner.specs,runner.rounds,1,'source','v4_source',False)
    selection['identity']=embedded.identity()
    path,frame=ns['final_paths'](embedded,selection)
    assert path.exists() and frame.shape==(2,337)
    assert frame.columns.tolist()==runner.data.sample_submission.columns.tolist()
    assert np.isfinite(frame.iloc[:,1:].to_numpy()).all()


def test_legacy_cache_requires_matching_provenance(tmp_path,monkeypatch):
    import importlib.metadata
    runner=make_runner(tmp_path,monkeypatch)
    directory=tmp_path/'artifacts/physics_features_v4/xgb_competition/example'
    directory.mkdir(parents=True)
    manifest=dict(raw=runner.raw,source='v4_source',member='xgb_competition',spec=runner.specs['xgb'],
        train=[0,1],predict=[2,3],rounds=runner.rounds['xgb'],tune=False,
        packages={p:importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','xgboost')})
    (directory/'manifest.json').write_text(json.dumps(manifest))
    for i,unit in enumerate(runner.data.units):
        np.savez_compressed(directory/f'unit_{i:02d}.npz',prediction=np.full((2,168),.4),rounds=runner.rounds['xgb'][unit])
    assert runner._legacy_cube('xgb',[0,1],[2,3]).shape==(2,2,168)
    runner.v4_source_digest='wrong'
    assert runner._legacy_cube('xgb',[0,1],[2,3]) is None
    runner.v4_source_digest='v4_source';runner.use_checkpoints=False
    assert runner._legacy_cube('xgb',[0,1],[2,3]) is None
