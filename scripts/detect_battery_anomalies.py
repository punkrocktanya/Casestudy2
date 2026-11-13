"""Retrospective, cell-specific candidate anomaly screening from Gold."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_capacity_fade import fit,predict

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/anomaly_detection'

def robust_z(s):
    center=s.median(); scale=1.4826*(s-center).abs().median()
    # Undefined scores are retained if there is no robust dispersion.
    return (s-center)/scale if scale>1e-12 else pd.Series(np.nan,index=s.index)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gold=pd.read_parquet(ROOT/'outputs/gold/battery_cycle_features_gold.parquet')
    frames=[]; sensitivity=[]
    figspace,space=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,(cell,g) in zip(space,gold.groupby('Dataset_ID')):
        g=g[g.Computationally_Complete & g.Discharge_Integration_Valid & g.Cycle_Index.between(2,100)].sort_values('Cycle_Index').copy().reset_index(drop=True)
        assert np.array_equal(g.Cycle_Index,np.arange(2,101))
        y=g.Discharge_Capacity_Ah; n=g.Cycle_Index.to_numpy()-2
        p=fit('Exponential',n,n,y.to_numpy())
        g['Expected_Ah']=predict('Exponential',p,n,n)
        g['Capacity_Residual_Ah']=y-g.Expected_Ah
        g['Exponential_Z']=robust_z(g.Capacity_Residual_Ah)
        # Adjacent interpolation removes a local linear trend; boundary cycles undefined.
        g['Local_Deviation_Ah']=y-(y.shift(1)+y.shift(-1))/2
        g['Local_Z']=robust_z(g.Local_Deviation_Ah)
        g['Capacity_Change_Pct']=y.pct_change(fill_method=None)*100
        local_change=g.Capacity_Change_Pct.rolling(11,center=True,min_periods=5).median()
        g['Change_Deviation_Pct']=g.Capacity_Change_Pct-local_change
        g['Residual_Detrended_Ah']=g.Capacity_Residual_Ah-g.Capacity_Residual_Ah.rolling(11,center=True,min_periods=5).median()
        g['CE_Deviation']=g.Coulombic_Efficiency_pct-g.Coulombic_Efficiency_pct.median()
        g['Exponential_Flag']=g.Exponential_Z.abs()>3
        g['Local_Flag']=g.Local_Z.abs()>3
        g['Trend_Candidate']=g.Exponential_Flag & g.Local_Flag
        g['CE_Above_100']=g.Coulombic_Efficiency_pct>100
        cols=['Residual_Detrended_Ah','Change_Deviation_Pct','CE_Deviation']
        valid=np.isfinite(g[cols]).all(axis=1)
        x=RobustScaler().fit_transform(g.loc[valid,cols])
        g['DBSCAN_Label']=pd.Series(pd.NA,index=g.index,dtype='Int64')
        g['DBSCAN_Outlier']=pd.Series(pd.NA,index=g.index,dtype='boolean')
        counts=np.zeros(len(x))
        for eps in [.5,.8,1.2,1.6]:
            for minimum in [4,5,7]:
                labels=DBSCAN(eps=eps,min_samples=minimum).fit_predict(x)
                counts+=labels==-1
                sensitivity.append(dict(Dataset_ID=cell,eps=eps,min_samples=minimum,outliers=int((labels==-1).sum()),clusters=len(set(labels)-{-1})))
                if eps==.8 and minimum==5:
                    g.loc[valid,'DBSCAN_Label']=labels
                    g.loc[valid,'DBSCAN_Outlier']=labels==-1
        g.loc[valid,'DBSCAN_Noise_Fraction']=counts/12
        g['Decision']='Not flagged by primary checks'
        g.loc[g.Local_Flag|g.Exponential_Flag,'Decision']='Residual/local review'
        g.loc[g.DBSCAN_Outlier.fillna(False),'Decision']='DBSCAN candidate'
        g.loc[g.Trend_Candidate & g.DBSCAN_Outlier.fillna(False),'Decision']='Both methods'
        g.loc[~valid,'Decision']='DBSCAN not evaluated: missing feature'
        g['Reason']=g.apply(lambda r:'; '.join(k for k,b in [('exponential residual',r.Exponential_Flag),('local deviation',r.Local_Flag),('CE above 100%',r.CE_Above_100)] if b),axis=1)
        frames.append(g)
        ax.scatter(x[:,0],x[:,1],c=g.loc[valid,'DBSCAN_Label'].astype(int),cmap='tab10',s=25)
        ax.set(title=str(cell),xlabel='Robust-scaled detrended residual',ylabel='Robust-scaled change deviation')
        fig,a=plt.subplots(figsize=(10,5),constrained_layout=True)
        a.plot(g.Cycle_Index,y,color='black',label='Observed'); a.plot(g.Cycle_Index,g.Expected_Ah,label='Exponential reference')
        for mask,label,marker in [(g.Local_Flag,'Local deviation','x'),(g.DBSCAN_Outlier.fillna(False),'DBSCAN noise','o')]:
            a.scatter(g.loc[mask,'Cycle_Index'],y[mask],label=label,marker=marker,s=55)
        a.set(title=str(cell),xlabel='Cycle',ylabel='Delivered capacity (Ah)'); a.legend()
        fig.savefig(OUT/f'anomaly_capacity_{cell}.png',dpi=160); plt.close(fig)
    figspace.suptitle('Two-dimensional projection of three-feature DBSCAN; -1 denotes noise')
    figspace.savefig(OUT/'dbscan_clusters.png',dpi=160); plt.close(figspace)
    result=pd.concat(frames,ignore_index=True)
    result.to_csv(OUT/'anomaly_features.csv',index=False)
    result[result.Exponential_Flag|result.Local_Flag].to_csv(OUT/'trend_anomalies.csv',index=False)
    result[['Dataset_ID','Cycle_Index','DBSCAN_Label','DBSCAN_Outlier','DBSCAN_Noise_Fraction']].to_csv(OUT/'dbscan_labels.csv',index=False)
    result[['Dataset_ID','Cycle_Index','Decision','Reason','Exponential_Z','Local_Z','DBSCAN_Noise_Fraction','CE_Above_100','Quality_Status']].to_csv(OUT/'anomaly_summary.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'parameter_sensitivity.csv',index=False)
    lines=['# Battery anomaly screening','',
        '- Source: Gold, complete valid cycles 2–100; cells fitted separately.',
        '- This is retrospective screening. Full-series fitting and centered neighborhoods use future observations; it is not an online warning system.',
        '- Exponential reference: Q=q0 exp(-kn). Residual = observed minus expected.',
        '- Robust z=(value−median)/(1.4826 MAD); |z|>3 triggers review. Zero MAD leaves scores undefined.',
        '- Local deviation = Q(n)−[Q(n−1)+Q(n+1)]/2. Endpoints have no local score. A single unusual cycle can also flag its neighbors.',
        '- Trend candidate requires both exponential and local flags, to reduce confusion with model curvature. Individual flags remain visible.',
        '- DBSCAN uses residual minus its centered 11-cycle median, percentage change minus its centered 11-cycle median, and CE minus its cell median. RobustScaler applies median/IQR scaling.',
        '- Missing features are excluded, not imputed. Cycle 2 has no prior selected capacity and is not evaluated by DBSCAN.',
        '- Primary eps=0.8, min_samples=5 are provisional. Twelve combinations of eps=0.5/0.8/1.2/1.6 and min_samples=4/5/7 provide noise fractions. These fractions are parameter sensitivity, not probabilities.',
        '- DVA and apparent DCIR are omitted: no validated complete-cycle feature set is established. Raw energy/capacity are omitted from clustering to limit degradation-trend confounding.',
        '- Cluster membership does not imply health. A noise point does not prove sensor error or battery damage. No records are deleted.',
        '- Existing quality status and CE flags are retained for cross-checking. Attribution to measurement, protocol or battery behaviour requires reviewing the source trace.',
        '', '## Results','']
    for cell,g in result.groupby('Dataset_ID'):
        lines.append(f'- {cell}: exponential flags {int(g.Exponential_Flag.sum())}; local flags {int(g.Local_Flag.sum())}; joint trend candidates {int(g.Trend_Candidate.sum())}; primary DBSCAN noise {int(g.DBSCAN_Outlier.sum())}; DBSCAN unevaluated {int(g.DBSCAN_Outlier.isna().sum())}.')
        candidates=g[g.Local_Flag|g.Exponential_Flag|g.DBSCAN_Outlier.fillna(False)]
        for _,r in candidates.iterrows(): lines.append(f'  - Cycle {r.Cycle_Index}: {r.Decision}; {r.Reason}; DBSCAN noise fraction={r.DBSCAN_Noise_Fraction:.2f}.')
    (OUT/'anomaly_detection_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[-20:]))

if __name__=='__main__': main()
