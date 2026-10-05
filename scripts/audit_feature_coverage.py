"""Inventory every v4 model input and saved gain/shift diagnostics. No training."""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.sintef_pipeline import build_project_data
from src.sintef_research_features import build_research_features
from src.sintef_feature_refinement import build_refinement_features


def main():
    data=build_project_data(ROOT)
    extra,local,_=build_research_features(data)
    additions,_=build_refinement_features(data)
    years=pd.DatetimeIndex(pd.concat([data.train_cases,data.test_cases]).starttime).year
    train=np.repeat(years<2022,168)
    comparisons={'audit_2022':np.repeat(years==2022,168),'test_2023_2024':np.repeat(years>=2023,168)}
    out=ROOT/'artifacts/prediction_gap_audit';out.mkdir(exist_ok=True)
    rows=[]
    common=pd.concat([data.X_all[data.feature_sets['full_engineered']],extra],axis=1)
    for plant in local:
        X=pd.concat([common,local[plant],additions['competition'][plant]],axis=1)
        a=X.to_numpy()[train]
        lo,hi=a.min(axis=0),a.max(axis=0)
        mu,sd=a.mean(axis=0,dtype=np.float64),a.std(axis=0,dtype=np.float64)
        for period,mask in comparisons.items():
            b=X.to_numpy()[mask]
            shift=np.abs(b.mean(axis=0,dtype=np.float64)-mu)/np.maximum(sd,1e-6)
            outside=((b<lo)|(b>hi)).mean(axis=0)
            for j,name in enumerate(X.columns):
                group=data.feature_group.get(name,'physics_global' if name in extra else 'physics_local' if name in local[plant] else 'price_competition')
                rows.append(dict(plant=plant,feature=name,group=group,period=period,
                    constant_before_2022=bool(sd[j]==0),missing=int(np.isnan(b[:,j]).sum()),
                    absolute_standardized_mean_shift=float(shift[j]),fraction_outside_past_range=float(outside[j])))
    inventory=pd.DataFrame(rows)
    inventory.to_csv(out/'all_v4_feature_diagnostics.csv',index=False)
    print(inventory.groupby(['group','period']).agg(feature_names=('feature','nunique'),mean_shift=('absolute_standardized_mean_shift','median'),outside=('fraction_outside_past_range','mean')).to_string())
    gains=[]
    for path in (ROOT/'artifacts/physics_features_v4').glob('*/*/manifest.json'):
        m=json.loads(path.read_text())
        if m['member'] not in ('xgb_competition','lgb_competition') or m['stage']!='audit_2022':continue
        for i,unit in enumerate(data.units):
            table=pd.read_csv(path.parent/f'importance_{i:02d}.csv')
            table['normalized_gain']=table.gain/max(table.gain.sum(),1e-30)
            table['unit']=unit;table['model']=m['member']
            gains.append(table)
    gain_table=pd.concat(gains,ignore_index=True)
    gain_table.to_csv(out/'v4_all_feature_gains_2022.csv',index=False)
    print(gain_table.groupby('feature').normalized_gain.mean().sort_values(ascending=False).head(25).to_string())
    print('All per-model feature rows:',len(gain_table),'Input inventory rows:',len(inventory))


if __name__=='__main__':main()
