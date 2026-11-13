"""Capacity-fade fits using bounded one-dimensional profile least squares (NumPy)."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/capacity_fade'
MODELS=['Linear','Exponential','Square-root time','Power law']

def profile(x,y):
    # y = intercept - slope*x; nonnegative slope and intercept.
    slope=max(0.,-float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)))
    intercept=max(0.,float(np.mean(y+slope*x)))
    return intercept,slope

def fit(model,n,t,y):
    if model=='Linear':
        q,a=profile(n,y); return [q,a]
    if model=='Square-root time':
        q,a=profile(np.sqrt(t),y); return [q,a]
    lo,hi=(0.,.2) if model=='Exponential' else (.05,2.)
    best=None
    for _ in range(5):
        grid=np.linspace(lo,hi,301)
        for z in grid:
            if model=='Exponential':
                v=np.exp(-z*n); q=max(0.,float(v@y/(v@v))); p=[q,z]; pred=q*v
            else:
                v=n**z; q,a=profile(v,y); p=[q,a,z]; pred=q-a*v
            loss=float(np.sum((y-pred)**2))
            if best is None or loss<best[0]: best=(loss,z,p)
        step=(hi-lo)/300; lo=max(grid[0],best[1]-step); hi=min(grid[-1],best[1]+step)
    return best[2]

def predict(model,p,n,t):
    if model=='Linear': return p[0]-p[1]*n
    if model=='Exponential': return p[0]*np.exp(-p[1]*n)
    if model=='Square-root time': return p[0]-p[1]*np.sqrt(t)
    return p[0]-p[1]*n**p[2]

def metrics(y,p):
    err=y-p
    return dict(MAE_Ah=float(np.abs(err).mean()),RMSE_Ah=float(np.sqrt(np.mean(err**2))),
                R2=float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),Negative_Predictions=int((p<0).sum()))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    goldpath=ROOT/'outputs/gold/battery_cycle_features_gold.parquet'
    gold=pd.read_parquet(goldpath)
    params=[]; scores=[]; predictions=[]; prepared=[]; lines=['# Cycle-by-cycle capacity fade','',
        '- Target: high-rate delivered discharge capacity (Ah), not certified SOH.',
        '- Primary analysis: cycles 2–100; incomplete boundary pairs excluded. REVIEW status is retained as provenance.',
        '- n = cycle index minus 2. t = days since cycle-2 discharge completion, using the test clock.',
        '- Retention = discharge Ah / cycle-2 discharge Ah x 100. Cycle 2 is not a verified beginning-of-life reference.',
        '- Four equations: linear Q=q0-a*n; exponential Q=q0*exp(-k*n); square-root Q=q0-a*sqrt(t); power law Q=q0-a*n^b.',
        '- q0 is fitted. Parameters are constrained nonnegative; power exponent is bounded to [0.05, 2], exponential k to [0, 0.2] per cycle.',
        '- Fits minimize squared error in Ah. Nonlinear parameters use a dense profile search with five local refinements; no log transformation of the target.',
        '- Full-series fits describe the observed trajectory. Separate chronological evaluation trains on 2–70, validates on 71–85, then refits on 2–85 and tests on 86–100.',
        '- Square-root-time evaluation uses observed future timestamps and is a conditional retrospective comparison, not a deployable fixed-origin forecast. It is excluded from selecting the cycle-only forecast winner.',
        '- Negative predictions are reported, not clipped. No extrapolated lifetime or mechanism identification is claimed.','']
    for ds,g in gold.groupby('Dataset_ID'):
        d=pd.read_parquet(next((ROOT/'outputs/silver').glob('*'+ds+'_silver.parquet')))
        end=d[d.Phase.eq('DISCHARGE')].groupby('Cycle_Index')['Test_Time(s)'].last()
        g=g[g.Computationally_Complete & g.Cycle_Index.between(2,100)].copy().sort_values('Cycle_Index')
        assert len(g)==99 and g.Discharge_Integration_Valid.all()
        g['n']=g.Cycle_Index-2; g['t_days']=(g.Cycle_Index.map(end)-end.loc[2])/86400
        g['Retention_pct']=g.Discharge_Capacity_Ah/g.Discharge_Capacity_Ah.iloc[0]*100
        g['Capacity_Change_Ah']=g.Discharge_Capacity_Ah.diff()
        g['Capacity_Change_pct']=g.Discharge_Capacity_Ah.pct_change()*100
        # Cycle-to-cycle fade rate.  The signed change follows ΔQ_n = Q_n-Q_(n-1);
        # negative values therefore mean capacity loss.  A centered rolling mean
        # makes the trend easier to see without changing the underlying values.
        g['Fade_Rate_Ah_per_Cycle']=-g['Capacity_Change_Ah']
        g['Fade_Rate_pct_per_Cycle']=-g['Capacity_Change_pct']
        g['Fade_Rate_Ah_per_Cycle_Smoothed']=g['Capacity_Change_Ah'].rolling(7,center=True,min_periods=3).mean()
        g['Fade_Rate_pct_per_Cycle_Smoothed']=g['Capacity_Change_pct'].rolling(7,center=True,min_periods=3).mean()
        prepared.append(g)
        n=g.n.to_numpy(float); t=g.t_days.to_numpy(float); y=g.Discharge_Capacity_Ah.to_numpy(float)
        train=g.Cycle_Index.le(70).to_numpy(); val=g.Cycle_Index.between(71,85).to_numpy(); dev=g.Cycle_Index.le(85).to_numpy(); test=g.Cycle_Index.ge(86).to_numpy()
        clean=g.Coulombic_Efficiency_pct.le(100).to_numpy()
        fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        axs[0,0].scatter(g.Cycle_Index,y,s=13,color='black',label='Observed')
        axs[0,1].plot(g.Cycle_Index,g.Retention_pct,color='black'); axs[0,1].set_ylabel('Retention relative to cycle 2 (%)')
        axs[1,1].scatter(g.Cycle_Index,g.Capacity_Change_pct,s=13); axs[1,1].axhline(0,color='grey',lw=.7); axs[1,1].set_ylabel('Change from previous cycle (%)')
        validation={}
        for model in MODELS:
            p=fit(model,n,t,y); pred=predict(model,p,n,t)
            sensitivity=fit(model,n[clean],t[clean],y[clean]); sens=predict(model,sensitivity,n,t)
            params.append(dict(Dataset_ID=ds,Model=model,Scope='full_series',Parameters=json.dumps(p),Sensitivity_Parameters=json.dumps(sensitivity),Max_Sensitivity_Change_Ah=float(np.max(abs(pred-sens)))))
            scores.append(dict(Dataset_ID=ds,Model=model,Scope='descriptive_full',**metrics(y,pred)))
            axs[0,0].plot(g.Cycle_Index,pred,label=model)
            axs[1,0].plot(g.Cycle_Index,y-pred,label=model)
            for label,tr,ev in [('validation',train,val),('test_refit',dev,test)]:
                pp=fit(model,n[tr],t[tr],y[tr]); yy=predict(model,pp,n[ev],t[ev]); mm=metrics(y[ev],yy)
                scores.append(dict(Dataset_ID=ds,Model=model,Scope=label,**mm))
                params.append(dict(Dataset_ID=ds,Model=model,Scope=label,Parameters=json.dumps(pp)))
                if label=='validation': validation[model]=mm['RMSE_Ah']
                for cycle,actual,estimate in zip(g.Cycle_Index[ev],y[ev],yy):
                    predictions.append(dict(Dataset_ID=ds,Model=model,Scope=label,Cycle_Index=int(cycle),Observed_Ah=actual,Predicted_Ah=estimate))
            for cycle,actual,estimate in zip(g.Cycle_Index,y,pred):
                predictions.append(dict(Dataset_ID=ds,Model=model,Scope='descriptive_full',Cycle_Index=int(cycle),Observed_Ah=actual,Predicted_Ah=estimate))
        for label,tr,ev in [('validation',train,val),('test_refit',dev,test)]:
            persistence=np.repeat(y[tr][-1],ev.sum())
            scores.append(dict(Dataset_ID=ds,Model='Fixed-origin persistence',Scope=label,**metrics(y[ev],persistence)))
        winner=min((m for m in MODELS if m!='Square-root time'),key=validation.get)
        lines += [f'**Cell {ds}**','',f'- Capacity: {y[0]:.6f} to {y[-1]:.6f} Ah; decline {100*(1-y[-1]/y[0]):.2f}%.',
            f'- Cycle-only model selected using validation RMSE: **{winner}**.',
            f'- Sensitivity analysis excludes {int((~clean).sum())} flagged CE cycles; the primary analysis retains them.',
            '- Coefficients and separate descriptive/validation/test metrics are in the accompanying CSV files.','']
        axs[0,0].set_ylabel('Discharge capacity (Ah)'); axs[0,0].legend(fontsize=8)
        axs[1,0].set_ylabel('Observed minus fitted (Ah)'); axs[1,0].axhline(0,color='grey',lw=.7)
        for ax in axs.flat: ax.set_xlabel('Cycle index'); ax.grid(alpha=.2)
        fig.suptitle(f'{ds}: high-rate delivered capacity — descriptive fits')
        fig.savefig(OUT/f'capacity_fade_{ds}.png',dpi=170); plt.close(fig)
        rate_fig, rate_ax = plt.subplots(figsize=(10,5), constrained_layout=True)
        rate_ax.scatter(g.Cycle_Index, g.Capacity_Change_Ah, s=14, alpha=.35, color='steelblue', label='ΔQ per cycle')
        rate_ax.plot(g.Cycle_Index, g.Fade_Rate_Ah_per_Cycle_Smoothed, color='darkred', lw=2,
                     label='7-cycle smoothed ΔQ')
        rate_ax.axhline(0, color='grey', lw=.8)
        rate_ax.set_title(f'{ds}: cycle-to-cycle capacity change and smoothed fade rate')
        rate_ax.set_xlabel('Cycle index'); rate_ax.set_ylabel('ΔQ = Qₙ − Qₙ₋₁ (Ah)')
        rate_ax.grid(alpha=.2); rate_ax.legend()
        rate_fig.savefig(OUT/f'capacity_fade_rate_{ds}.png', dpi=170); plt.close(rate_fig)
    rate_table=pd.concat(prepared, ignore_index=True)
    rate_table.to_csv(OUT/'cycle_capacity_analysis.csv',index=False)
    rate_table[['Dataset_ID','Cycle_Index','Discharge_Capacity_Ah','Capacity_Change_Ah',
                'Capacity_Change_pct','Fade_Rate_Ah_per_Cycle','Fade_Rate_pct_per_Cycle',
                'Fade_Rate_Ah_per_Cycle_Smoothed','Fade_Rate_pct_per_Cycle_Smoothed']].to_csv(
                    OUT/'capacity_fade_rate_analysis.csv', index=False)
    pd.DataFrame(params).to_csv(OUT/'model_parameters.csv',index=False)
    pd.DataFrame(scores).to_csv(OUT/'evaluation_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'predictions.csv',index=False)
    lines+=['**Interpretation limits**','',
        '- Cycle-to-cycle change is ΔQ_n = Q_n − Q_(n−1). The plotted smoothed line is a centred 7-cycle rolling mean; negative ΔQ means fading. The separate positive fade-rate columns use −ΔQ.','',
        '- High-current polarization, changing charge endpoints and irreversible aging can all contribute. A fitted exponent does not identify their individual contributions.',
        '- Missing temperature and reference-capacity tests prevent certified SOH or electrode-mechanism claims.',
        '- Models are fitted independently to two trajectories; results do not establish population-level generalization.',
        '- The last 15 cycles span a narrow capacity range, so test R2 may be poor even when Ah errors look small.',
        '- Units of a: Ah/cycle for linear; Ah/sqrt(day) for square-root; Ah/cycle^b for power law. k is per cycle.','']
    (OUT/'capacity_fade_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'run_manifest.json').write_text(json.dumps({'gold_sha256':hashlib.sha256(goldpath.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':198,'target':'high-rate delivered discharge Ah'},indent=2))
    print(pd.DataFrame(scores).to_string(index=False))

if __name__=='__main__': main()
