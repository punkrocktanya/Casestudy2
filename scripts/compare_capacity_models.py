"""Reproducible late-cycle comparison; separate from the earlier reviewed package."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from individual_curve_rul import INPUT
from estimate_rul_xgboost_shared import examples, FEATURES
from estimate_rul_lstm_minimal import train, torch
from fit_capacity_fade import fit, predict, metrics

OUT=INPUT.parents[1]/'capacity_comparison_local'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(INPUT,dtype={'Dataset_ID':str})
    records,scores=[],[]
    for cell,g in data.groupby('Dataset_ID'):
        g=g.sort_values('Cycle_Index')
        for stage,origin in [('validation',70),('test',85)]:
            h=g[g.Cycle_Index<=origin]; target=g[g.Cycle_Index.between(origin+1,origin+15)]
            n=h.Cycle_Index.to_numpy()-2; y=h.Discharge_Capacity_Ah.to_numpy(); future=target.Cycle_Index.to_numpy()-2
            exp=fit('Exponential',n,n,y)
            bounds=([1e-7,0,0],[10,2,2])
            def residual(p):return p[0]*np.exp(-p[1]*np.sqrt(n)-p[2]*n)-y
            fits=[least_squares(residual,[y[0],a,b],bounds=bounds) for a,b in [(.01,.01),(.1,.001),(.001,.03)]]
            converged=[f for f in fits if f.success]
            if not converged:raise RuntimeError('Physics-surrogate fit failed')
            phy=min(converged,key=lambda f:np.sum(f.fun**2)).x
            tab=examples(h); rf=RandomForestRegressor(n_estimators=200,max_depth=4,min_samples_leaf=3,random_state=42,n_jobs=1)
            rf.fit(tab[FEATURES],tab.Target_Change)
            q=h.Discharge_Capacity_Ah.to_numpy()/y[0]; w=3
            xx=np.array([q[j-w:j] for j in range(w,len(q))],dtype=np.float32)[...,None]
            yy=np.diff(q)[w-1:].astype(np.float32)
            model,_=train(xx,yy,42)
            trajectories={'Exponential':predict('Exponential',exp,future,future),
                          'Physics-constrained surrogate':phy[0]*np.exp(-phy[1]*np.sqrt(future)-phy[2]*future),
                          'Persistence':np.repeat(y[-1],15)}
            from estimate_rul_xgboost_shared import features
            for name in ['Random Forest','LSTM']:
                history=list(q); predicted=[]
                for _ in range(15):
                    if name=='Random Forest':delta=float(rf.predict(pd.DataFrame([features(history)],columns=FEATURES))[0])
                    else:
                        with torch.inference_mode():delta=float(model(torch.tensor(history[-3:],dtype=torch.float32).reshape(1,3,1)).item())
                    history.append(history[-1]+delta);predicted.append(history[-1]*y[0])
                trajectories[name]=np.array(predicted)
            for name,values in trajectories.items():
                scores.append(dict(Dataset_ID=cell,Stage=stage,Model=name,**metrics(target.Discharge_Capacity_Ah.to_numpy(),values)))
                records.extend(dict(Dataset_ID=cell,Stage=stage,Model=name,Cycle_Index=int(c),Observed_Ah=float(a),Predicted_Ah=float(b)) for c,a,b in zip(target.Cycle_Index,target.Discharge_Capacity_Ah,values))
    pd.DataFrame(scores).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(records).to_csv(OUT/'predictions.csv',index=False)
    (OUT/'settings.json').write_text(json.dumps(dict(window=3,lstm_seed=42,forecast='fixed-origin recursive 15 cycles',physics='positive nonincreasing phenomenological surrogate; not mechanistic'),indent=2))
    print(pd.DataFrame(scores).to_string(index=False))


if __name__=='__main__':main()
