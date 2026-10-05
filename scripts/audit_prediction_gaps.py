"""Read saved forward predictions; small linear probes only, never base-model training."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.sintef_pipeline import build_project_data


def load_saved(member, stage):
    matches = []
    for path in (ROOT/'artifacts/physics_features_v4'/member).glob('*/manifest.json'):
        m = json.loads(path.read_text())
        if m['stage'] == stage and len(list(path.parent.glob('unit_*.npz'))) == 14:
            matches.append((path, m))
    if len(matches) != 1:
        raise ValueError(f'Ambiguous/missing cache: {member}/{stage}')
    path, m = matches[0]
    cube = np.stack([np.load(path.parent/f'unit_{i:02d}.npz')['prediction'] for i in range(14)], axis=1)
    return np.array(m['predict']), cube


def contexts(x, l, units, i):
    p = (x+l)*.5
    z = logit(np.clip(p, 1e-5, 1-1e-5))
    own = z[:, i]
    cols = [logit(np.clip(x[:, i], 1e-5, 1-1e-5)), logit(np.clip(l[:, i], 1e-5, 1-1e-5))]
    for offset in (-24, -6, -3, -1, 1, 3, 6, 24):
        cols.append(own[:, np.clip(np.arange(168)+offset, 0, 167)] - own)
    cols += [np.broadcast_to(own.mean(axis=1)[:, None], own.shape)-own]
    siblings = [j for j, u in enumerate(units) if u.rsplit('_G',1)[0] == units[i].rsplit('_G',1)[0] and j != i]
    cols += [z[:, siblings].mean(axis=1) if siblings else np.zeros_like(own)]
    return np.column_stack([v.ravel() for v in cols])


def main():
    data = build_project_data(ROOT)
    ids1, x1 = load_saved('xgb_competition', 'ablation_2021')
    ids2, x2 = load_saved('xgb_competition', 'audit_2022')
    for year, ids in [(2021, ids1), (2022, ids2)]:
        other, _ = load_saved('lgb_competition', 'ablation_2021' if year==2021 else 'audit_2022')
        np.testing.assert_array_equal(ids, other)
    _, l1 = load_saved('lgb_competition', 'ablation_2021')
    _, l2 = load_saved('lgb_competition', 'audit_2022')
    y1, y2 = data.target_cube[ids1], data.target_cube[ids2]
    # Exclude 2021 cases with horizon overlapping the first 2022 validation case.
    starts = pd.DatetimeIndex(data.train_cases.starttime)
    keep = starts[ids1] + pd.Timedelta(hours=167) < starts[ids2].min()
    base = (x2+l2)*.5
    predictions = {'baseline_v4': base, 'platt_per_unit': np.empty_like(base), 'linear_context': np.empty_like(base)}
    details = []
    with threadpool_limits(limits=1):
        for i, unit in enumerate(data.units):
            own = y2[:, i]
            transition = np.zeros_like(own, dtype=bool)
            transition[:,1:] = own[:,1:] != own[:,:-1]
            near = transition.copy()
            for k in (1,2):
                near[:,k:] |= transition[:,:-k]; near[:,:-k] |= transition[:,k:]
            err = (base[:,i]-own)**2
            details.append(dict(unit=unit, auc=roc_auc_score(own.ravel(),base[:,i].ravel()),
                brier=err.mean(), on_rate=own.mean(), predicted_rate=base[:,i].mean(),
                transition_near_fraction=near.mean(), transition_error_share=err[near].sum()/err.sum(),
                constant_case_fraction=(own.std(axis=1)==0).mean()))
            for name in ('platt_per_unit', 'linear_context'):
                if name == 'platt_per_unit':
                    a = logit(np.clip((x1[:,i]+l1[:,i])*.5,1e-5,1-1e-5)).reshape(-1,1)
                    b = logit(np.clip(base[:,i],1e-5,1-1e-5)).reshape(-1,1)
                else:
                    a, b = contexts(x1,l1,data.units,i), contexts(x2,l2,data.units,i)
                a = a.reshape(len(ids1),168,-1)[keep].reshape(-1,a.shape[1])
                scale = StandardScaler().fit(a)
                model = LogisticRegression(C=.1, max_iter=100, solver='lbfgs').fit(scale.transform(a),y1[keep,i].ravel())
                predictions[name][:,i] = model.predict_proba(scale.transform(b))[:,1].reshape(len(ids2),168)
    out = ROOT/'artifacts/prediction_gap_audit'; out.mkdir(exist_ok=True)
    pd.DataFrame(details).to_csv(out/'unit_gaps_2022.csv',index=False)
    scores = [dict(model=n, auc=roc_auc_score(y2.ravel(),p.ravel())) for n,p in predictions.items()]
    pd.DataFrame(scores).to_csv(out/'linear_probes_2022.csv',index=False)
    print(pd.DataFrame(scores).to_string(index=False))
    print(pd.DataFrame(details).sort_values('brier',ascending=False).to_string(index=False))
    for block in range(7):
        sl=slice(block*24,(block+1)*24)
        print('horizon day',block+1,'AUC',roc_auc_score(y2[:,:,sl].ravel(),base[:,:,sl].ravel()))
    for month in range(1,13):
        use=starts[ids2].month==month
        print('month',month,'AUC',roc_auc_score(y2[use].ravel(),base[use].ravel()))

    # One small case-level (not case-hour) multi-output forest: ~2,500 examples.
    # A diagnostic, not the final model or a hyperparameter search.
    from sklearn.ensemble import ExtraTreesRegressor
    X = data.X_all.to_numpy(np.float32).reshape(data.n_all_cases,168,-1)
    summary = np.column_stack([X.mean(axis=1), X.std(axis=1), X[:,0], X[:,-1]])
    train = np.flatnonzero(starts + pd.Timedelta(hours=167) < starts[ids2].min())
    model = ExtraTreesRegressor(n_estimators=32,max_depth=12,min_samples_leaf=5,max_features=.7,n_jobs=1,random_state=42)
    model.fit(summary[train], data.target_cube[train].mean(axis=2))
    weekly = model.predict(summary[ids2])
    z = logit(np.clip(base,1e-5,1-1e-5))
    budget_rows=[]
    for name, target in [('small_forest_budget',weekly), ('ORACLE_true_weekly_fraction_DIAGNOSTIC_ONLY',y2.mean(axis=2))]:
        lo=np.full(target.shape,-20.); hi=-lo
        for _ in range(30):
            mid=(lo+hi)*.5
            low=expit(z+mid[:,:,None]).mean(axis=2)<target
            lo=np.where(low,mid,lo); hi=np.where(low,hi,mid)
        shift=(lo+hi)*.5
        corrected=expit(z+shift[:,:,None]*(.25 if name=='small_forest_budget' else 1.))
        budget_rows.append(dict(model=name,auc=roc_auc_score(y2.ravel(),corrected.ravel()),
            weekly_mae=np.abs(target-y2.mean(axis=2)).mean()))
    budget_rows.append(dict(model='baseline_weekly_fraction',auc=roc_auc_score(y2.ravel(),base.ravel()),weekly_mae=np.abs(base.mean(axis=2)-y2.mean(axis=2)).mean()))
    pd.DataFrame(budget_rows).to_csv(out/'weekly_budget_probe.csv',index=False)
    print(pd.DataFrame(budget_rows).to_string(index=False))


if __name__ == '__main__':
    main()
