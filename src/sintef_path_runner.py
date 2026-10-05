"""Bounded staged boosting experiment. Importing this module never trains."""
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from src.sintef_pipeline import HORIZON, rows_for_cases, make_submission
from src.sintef_challenger_runner import raw_digest, save_json, purged_year, score_cube, fit_member


def factor_key(factor):
    return f'{float(factor):g}'


def scaled_rounds(original, factor):
    if not 0 < float(factor) <= 2:
        raise ValueError('Round multiplier must be in (0, 2]')
    return max(1, int(round(int(original)*float(factor))))


def checked_cube(values, cases, units):
    values=np.asarray(values,dtype=np.float32)
    if values.shape != (cases,units,HORIZON) or not np.isfinite(values).all():
        raise ValueError('Invalid prediction dimensions or non-finite values')
    if values.min()<0 or values.max()>1:
        raise ValueError('Prediction outside [0,1]')
    return values


def staged_predict(model, backend, X, rounds, threads):
    if backend=='xgboost':
        return model.predict_proba(X,iteration_range=(0,int(rounds)))[:,1]
    return model.predict(X,num_iteration=int(rounds),num_threads=threads)


class PathRunner:
    def __init__(self,data,extra,local,competition,paths,specs,rounds,threads,
                 source_digest,v4_source_digest,use_checkpoints=True):
        self.data,self.local,self.competition,self.paths=data,local,competition,paths
        self.specs,self.rounds=specs,rounds
        self.threads=max(1,min(int(threads),os.cpu_count() or 1))
        self.use_checkpoints=bool(use_checkpoints)
        self.source_digest,self.v4_source_digest=source_digest,v4_source_digest
        self.base_names=list(data.feature_sets['full_engineered'])+list(extra.columns)
        self.base=np.column_stack([data.X_all[data.feature_sets['full_engineered']].to_numpy(np.float32),extra.to_numpy(np.float32)])
        self.raw=raw_digest(data.root)
        self.directory=data.root/'artifacts/multiscale_boosting_v5'
        self.directory.mkdir(parents=True,exist_ok=True)

    def identity(self):
        return dict(source=self.source_digest,raw=self.raw,specs=self.specs,rounds=self.rounds,
                    packages={p:importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','xgboost','lightgbm')})

    def _legacy_cube(self,family,train,predict):
        """Reuse ONLY exact v4-competition runs; no guessing from filenames alone."""
        if not self.use_checkpoints:
            return None
        member=family+'_competition'
        for path in sorted((self.data.root/'artifacts/physics_features_v4'/member).glob('*/manifest.json')):
            m=json.loads(path.read_text())
            spec=self.specs[family]
            wanted_packages={p:importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn',spec['backend'])}
            if not (m.get('raw')==self.raw and m.get('source')==self.v4_source_digest
                    and m.get('member')==member and m.get('spec')==spec
                    and m.get('train')==list(map(int,train)) and m.get('predict')==list(map(int,predict))
                    and m.get('rounds')==self.rounds[family] and not m.get('tune')
                    and m.get('packages')==wanted_packages):
                continue
            pieces=[]
            for i,unit in enumerate(self.data.units):
                checkpoint=path.parent/f'unit_{i:02d}.npz'
                if not checkpoint.exists():
                    break
                with np.load(checkpoint,allow_pickle=False) as saved:
                    if int(saved['rounds']) != self.rounds[family][unit]:
                        break
                    pieces.append(saved['prediction'].copy())
            if len(pieces)==len(self.data.units):
                cube=checked_cube(np.stack(pieces,axis=1),len(predict),len(self.data.units))
                save_json(self.directory/f'imported_{family}_{int(predict[0])}.json',dict(
                    path=str(path.relative_to(self.data.root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source=m['source'],raw=self.raw,note='Matching notebook-17 competition predictions; thread counts may differ.'))
                print(f'Restored exact v4 {family}: {path.parent}',flush=True)
                return cube
        return None

    def run(self,family,variant,train,predict,stage,factors=(1.,1.5,2.)):
        if family not in self.specs or variant not in ('v4','path'):
            raise ValueError('Unknown model or feature variant')
        train,predict=np.asarray(train,dtype=int),np.asarray(predict,dtype=int)
        if not len(train) or not len(predict) or len(np.intersect1d(train,predict)):
            raise ValueError('Empty or overlapping training/prediction cases')
        if np.any(train<0) or np.any(train>=self.data.n_train_cases) or np.any(predict<0) or np.any(predict>=self.data.n_all_cases):
            raise ValueError('Case index outside dataset')
        factors=tuple(sorted(set(float(f) for f in factors)))
        if not factors:
            raise ValueError('No requested tree counts')
        for f in factors: scaled_rounds(1,f)
        if variant=='v4' and factors==(1.,):
            restored=self._legacy_cube(family,train,predict)
            if restored is not None:
                return {'1':restored}
        manifest=dict(self.identity(),family=family,variant=variant,stage=stage,threads=self.threads,
                      train=list(map(int,train)),predict=list(map(int,predict)),factors=list(factors))
        signature=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:24]
        directory=self.directory/f'{stage}_{family}_{variant}'/signature
        directory.mkdir(parents=True,exist_ok=True)
        save_json(directory/'manifest.json',manifest)
        cubes={factor_key(f):np.empty((len(predict),len(self.data.units),HORIZON),np.float32) for f in factors}
        train_rows,pred_rows=rows_for_cases(train),rows_for_cases(predict)
        timings=[]
        for i,unit in enumerate(self.data.units):
            start=time.perf_counter()
            counts=[scaled_rounds(self.rounds[family][unit],f) for f in factors]
            checkpoint=directory/f'unit_{i:02d}.npz'
            if self.use_checkpoints and checkpoint.exists():
                with np.load(checkpoint,allow_pickle=False) as saved:
                    if saved['rounds'].tolist()!=counts:
                        raise ValueError(f'Mismatched checkpoint: {checkpoint}')
                    predictions=saved['predictions'].copy()
                status='restored'
            else:
                plant=unit.rsplit('_G',1)[0]
                parts=[self.base,self.local[plant].to_numpy(np.float32),self.competition[plant].to_numpy(np.float32)]
                names=self.base_names+list(self.local[plant].columns)+list(self.competition[plant].columns)
                if variant=='path':
                    parts.append(self.paths[plant].to_numpy(np.float32)); names+=list(self.paths[plant].columns)
                if len(set(names))!=len(names):
                    raise ValueError('Duplicate model features')
                X=np.ascontiguousarray(np.column_stack(parts),dtype=np.float32)
                del parts
                y=self.data.target_cube[train,i].ravel()
                print(f'{stage} | {family}/{variant} | {unit} | {max(counts)} trees; snapshots {counts}',flush=True)
                if len(np.unique(y))==1:
                    predictions=np.full((len(factors),len(predict),HORIZON),float(y[0]),np.float32)
                else:
                    with threadpool_limits(limits=self.threads),threadpool_limits(limits=1,user_api='blas'):
                        model,_=fit_member(self.specs[family],X[train_rows],y,max(counts),self.threads)
                        pred_X=X[pred_rows]
                        # One fit, three prefixes. No early stopping on the audit labels.
                        predictions=np.stack([staged_predict(model,self.specs[family]['backend'],pred_X,count,self.threads).reshape(len(predict),HORIZON) for count in counts]).astype(np.float32)
                        gain=model.feature_importances_ if self.specs[family]['backend']=='xgboost' else model.feature_importance(importance_type='gain')
                    pd.DataFrame(dict(feature=names,gain=gain)).sort_values('gain',ascending=False).to_csv(directory/f'importance_{i:02d}.csv',index=False)
                    del model,pred_X
                temporary=checkpoint.with_suffix('.tmp.npz')
                np.savez_compressed(temporary,predictions=predictions,rounds=np.asarray(counts))
                os.replace(temporary,checkpoint)
                del X,y
                gc.collect()
                status='trained'
            if predictions.shape!=(len(factors),len(predict),HORIZON) or not np.isfinite(predictions).all() or predictions.min()<0 or predictions.max()>1:
                raise ValueError(f'Invalid checkpoint: {checkpoint}')
            for k,f in enumerate(factors): cubes[factor_key(f)][:,i]=predictions[k]
            timings.append(dict(unit=unit,max_rounds=max(counts),seconds=time.perf_counter()-start,status=status))
            pd.DataFrame(timings).to_csv(directory/'runtime.csv',index=False)
            print(f'  {status}: {timings[-1]["seconds"]:.1f}s ({i+1}/{len(self.data.units)})',flush=True)
        return cubes


def select_pair(scores,predictions,truth):
    """Small, global search on development only; ties prefer existing features/counts."""
    chosen={}
    for family in ('xgb','lgb'):
        candidates=[row for row in scores if row['family']==family]
        best=max(candidates,key=lambda r:(r['micro_auc'],r['variant']=='v4',-r['factor']))
        chosen[family]=dict(variant=best['variant'],factor=best['factor'])
    x=predictions[('xgb',chosen['xgb']['variant'],factor_key(chosen['xgb']['factor']))]
    l=predictions[('lgb',chosen['lgb']['variant'],factor_key(chosen['lgb']['factor']))]
    grid=[dict(xgb_weight=w,micro_auc=score_cube(truth,w*x+(1-w)*l)) for w in (.25,.5,.75)]
    best=max(grid,key=lambda r:(r['micro_auc'],-abs(r['xgb_weight']-.5)))
    return dict(members=chosen,xgb_weight=best['xgb_weight'],development_auc=best['micro_auc']),grid


def develop_paths(runner):
    train,valid=purged_year(runner.data,2021)
    predictions,rows={},[]
    for family in ('xgb','lgb'):
        for variant in ('v4','path'):
            snapshots=runner.run(family,variant,train,valid,'development_2021')
            for factor,p in snapshots.items():
                predictions[(family,variant,factor)]=p
                row=dict(year=2021,family=family,variant=variant,factor=float(factor),micro_auc=score_cube(runner.data.target_cube[valid],p))
                rows.append(row)
                print(row,flush=True)
                pd.DataFrame(rows).to_csv(runner.directory/'development_scores.csv',index=False)
    selection,grid=select_pair(rows,predictions,runner.data.target_cube[valid])
    selection['identity']=runner.identity()
    selection['development_baseline_auc']=score_cube(runner.data.target_cube[valid],.5*(predictions[('xgb','v4','1')]+predictions[('lgb','v4','1')]))
    save_json(runner.directory/'selection.json',selection)
    pd.DataFrame(grid).to_csv(runner.directory/'blend_grid_2021.csv',index=False)
    return selection,pd.DataFrame(rows)


def diagnostic_tables(data,cases,predictions):
    """Descriptive per-unit/month/horizon errors; never used for unit-specific tuning."""
    rows=[]
    truth=data.target_cube[cases]
    months=pd.DatetimeIndex(data.train_cases.iloc[cases].starttime).month
    for name,p in predictions.items():
        for group,labels in [('unit',list(data.units)),('month',list(range(1,13))),('horizon_day',list(range(1,8)))]:
            for i,label in enumerate(labels):
                if group=='unit': y,q=truth[:,i],p[:,i]
                elif group=='month': y,q=truth[months==label],p[months==label]
                else: y,q=truth[:,:,(label-1)*24:label*24],p[:,:,(label-1)*24:label*24]
                if y.size:
                    auc=float(roc_auc_score(y.ravel(),q.ravel())) if len(np.unique(y))==2 else np.nan
                    rows.append(dict(model=name,group=group,label=label,auc=auc,brier=float(np.mean((q-y)**2)),on_rate=float(y.mean()),predicted_rate=float(q.mean())))
    return pd.DataFrame(rows)


def audit_paths(runner,selection):
    if selection.get('identity')!=runner.identity():
        raise ValueError('Selection configuration changed; rerun development')
    train,valid=purged_year(runner.data,2022)
    chosen,baseline={},{}
    for family in ('xgb','lgb'):
        member=selection['members'][family]
        if member['variant']=='v4':
            snapshots=runner.run(family,'v4',train,valid,'audit_2022',factors=(1.,member['factor']))
            baseline[family]=snapshots['1'];chosen[family]=snapshots[factor_key(member['factor'])]
        else:
            baseline[family]=runner.run(family,'v4',train,valid,'audit_2022',factors=(1.,))['1']
            chosen[family]=runner.run(family,'path',train,valid,'audit_2022',factors=(member['factor'],))[factor_key(member['factor'])]
    weight=selection['xgb_weight']
    candidate=weight*chosen['xgb']+(1-weight)*chosen['lgb']
    control=.5*(baseline['xgb']+baseline['lgb'])
    new_score,old_score=score_cube(runner.data.target_cube[valid],candidate),score_cube(runner.data.target_cube[valid],control)
    selection=dict(selection,audit_auc=new_score,audit_baseline_auc=old_score,audit_gain=new_score-old_score,passed_audit=bool(new_score>old_score))
    save_json(runner.directory/'selection.json',selection)
    np.savez_compressed(runner.directory/'audit_predictions.npz',case_ids=valid,candidate=candidate,baseline=control)
    diagnostic_tables(runner.data,valid,dict(candidate=candidate,baseline=control)).to_csv(runner.directory/'audit_diagnostics.csv',index=False)
    print(f'2022: candidate={new_score:.6f}; v4 baseline={old_score:.6f}; delta={new_score-old_score:+.6f}',flush=True)
    print('Historical audit improved.' if selection['passed_audit'] else 'NO confirmed improvement. Keep the existing Kaggle-best CSV as a fallback.',flush=True)
    return selection


def final_paths(runner,selection,filename='multiscale_boosting_v5.csv'):
    if selection.get('identity')!=runner.identity():
        raise ValueError('Selection configuration/source/data/packages changed')
    train=np.arange(runner.data.n_train_cases)
    test=np.arange(runner.data.n_train_cases,runner.data.n_all_cases)
    blend=np.zeros((len(test),len(runner.data.units),HORIZON),np.float32)
    for family in ('xgb','lgb'):
        member=selection['members'][family]
        p=runner.run(family,member['variant'],train,test,'final',factors=(member['factor'],))[factor_key(member['factor'])]
        weight=selection['xgb_weight'] if family=='xgb' else 1-selection['xgb_weight']
        blend+=weight*p
        path,_=make_submission(runner.data,p,f'{family}_multiscale_v5.csv')
        print('Single model:',path,flush=True)
    path,submission=make_submission(runner.data,checked_cube(blend,len(test),len(runner.data.units)),filename)
    print('SUBMISSION READY:',path,flush=True)
    return path,submission
