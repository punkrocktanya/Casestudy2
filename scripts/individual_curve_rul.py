"""Individual battery curve RUL models with a shared power-law exponent."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_capacity_fade import fit,predict,profile
ROOT=Path(__file__).resolve().parent.parent
INPUT=ROOT/'outputs/capacity_fade/cycle_capacity_analysis.csv'
OUT=ROOT/'outputs/rul_individual_curves'
MODELS=['Linear','Exponential','Square-root cycle','Power law']
CHECKPOINTS=[5,8,10,12,15,20]

def validate(g,checkpoint):
 g=g[g.Cycle_Index<=checkpoint]
 g=g[g.Computationally_Complete & g.Discharge_Integration_Valid].sort_values('Cycle_Index')
 if len(g)==0 or g.Cycle_Index.iloc[0]!=2 or g.Cycle_Index.iloc[-1]!=checkpoint or not np.all(np.diff(g.Cycle_Index)==1):raise ValueError('Incomplete/gapped observed history')
 return g

def fit_cell(g,model,shared_b=None):
 n=g.Cycle_Index.to_numpy()-2;q=g.Discharge_Capacity_Ah.to_numpy(float);scale=q[0];y=q/scale
 if model=='Linear':
  intercept,slope=profile(n,y);return [intercept,slope]
 if model=='Exponential':return fit('Exponential',n,n,y)
 if model=='Square-root cycle':
  intercept,slope=profile(np.sqrt(n),y);return [intercept,slope]
 if shared_b is None:raise ValueError('Shared power exponent required')
 x=n**shared_b;slope=max(0.,-np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2));intercept=max(0.,float(np.mean(y+slope*x)));return [intercept,slope,shared_b]

def curve(model,p,n):
 if model=='Exponential':return p[0]*np.exp(-p[1]*n)
 if model=='Square-root cycle':return p[0]-p[1]*np.sqrt(n)
 if model=='Power law':return p[0]-p[1]*n**p[2]
 return p[0]-p[1]*n

def main():
 OUT.mkdir(parents=True,exist_ok=True);d=pd.read_csv(INPUT,dtype={'Dataset_ID':str})
 groups={c:g[g.Computationally_Complete & g.Discharge_Integration_Valid].sort_values('Cycle_Index') for c,g in d.groupby('Dataset_ID')}
 eols={c:int(g.loc[g.Discharge_Capacity_Ah<.8*g.Discharge_Capacity_Ah.iloc[0],'Cycle_Index'].min()) for c,g in groups.items()}
 checkpoints=[c for c in CHECKPOINTS if sum(c<e for e in eols.values())>0]
 rows=[];traces=[];training=[]
 for cp in checkpoints:
  histories={c:validate(g,cp) for c,g in groups.items() if cp<eols[c]}
  # Estimate a common power exponent only from batteries still pre-EOL at this checkpoint.
  bgrid=np.linspace(.05,2,1000);losses=[]
  for b in bgrid:
   loss=0.
   for g in histories.values():
    n=g.Cycle_Index.to_numpy()-2;q=g.Discharge_Capacity_Ah.to_numpy(float);y=q/q[0];x=n**b;sl=max(0.,-np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2));it=max(0.,float(np.mean(y+sl*x)));loss+=float(np.sum((y-(it-sl*x))**2))
   losses.append(loss)
  shared_b=float(bgrid[int(np.argmin(losses))])
  params={c:{m:fit_cell(g,m,shared_b) for m in MODELS} for c,g in histories.items()}
  for c,g in histories.items():
   scale=float(g.Discharge_Capacity_Ah.iloc[0]);n=g.Cycle_Index.to_numpy()-2;y=g.Discharge_Capacity_Ah.to_numpy(float)/scale
   training.extend(dict(Dataset_ID=c,Checkpoint=cp,Cycle=int(cy),Normalized_Capacity=float(cap)) for cy,cap in zip(g.Cycle_Index,y))
   for m,p in params[c].items():
    current=float(curve(m,p,np.array([cp-2]))[0]);future=np.arange(cp+1,cp+1001);vals=curve(m,p,future-2)
    status='crossing_predicted';hit=np.flatnonzero(vals<.8);cross=int(future[hit[0]]) if len(hit) else None
    if current<.8:cross=None;status='fitted_crossing_at_or_before_origin'
    elif cross is None:status='no_crossing_within_horizon'
    ar=eols[c]-cp;pr=cross-cp if cross is not None else np.nan;err=pr-ar
    rows.append(dict(Dataset_ID=c,Model=m,Checkpoint=cp,Training_cycles=len(g),Actual_EOL=eols[c],Predicted_EOL=cross,Actual_RUL=ar,Predicted_RUL=pr,Error=err,Absolute_Error=abs(err),Status=status,Parameters=json.dumps(p),Shared_power_exponent=shared_b if m=='Power law' else np.nan))
    for cy,v in zip(future[:250],vals[:250]):traces.append(dict(Dataset_ID=c,Model=m,Checkpoint=cp,Forecast_Cycle=int(cy),Predicted_Capacity_Ah=float(v*scale)))
  fig,axes=plt.subplots(2,2,figsize=(12,9))
  for ax,m in zip(axes.flat,MODELS):
   for c,g in histories.items():
    scale=g.Discharge_Capacity_Ah.iloc[0];color='#146c94' if c=='001_2' else '#7c4bad';norm=g.Discharge_Capacity_Ah/scale
    ax.plot(g.Cycle_Index,g.Discharge_Capacity_Ah,'o-',ms=3,color=color,label=f'{c} measured')
    p=params[c][m];xx=np.arange(2,cp+1);ax.plot(xx,curve(m,p,xx-2)*scale,color=color,lw=2,label=f'{c} fitted')
    ff=np.arange(cp, min(55,eols[c]+8));ax.plot(ff,curve(m,p,ff-2)*scale,color=color,ls=':',lw=2,label=f'{c} forecast')
    ax.axhline(.8*scale,color=color,ls='--',alpha=.65)
   ax.axvline(cp,color='grey',ls=':');ax.set(title=m,xlabel='Cycle number',ylabel='Discharge capacity (Ah)',xlim=(2,55));ax.grid(alpha=.16)
   ax.text(.02,.97,{'Linear':'Each cell has its own Ah loss per cycle.','Exponential':'Each cell has its own proportional fade rate.','Square-root cycle':'Each cell has its own loss coefficient.','Power law':f'Cell-specific loss; shared exponent b={shared_b:.3f}.'}[m],transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
  h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=8);fig.suptitle(f'Individual battery curve predictions using observations through cycle {cp}');fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(OUT/f'checkpoint_{cp}.png',dpi=170);plt.close(fig)
 result=pd.DataFrame(rows);result.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(traces).to_csv(OUT/'forecast_traces.csv',index=False);pd.DataFrame(training).to_csv(OUT/'training_history.csv',index=False)
 metrics=result.groupby(['Dataset_ID','Model']).agg(Checkpoints=('Checkpoint','size'),Scored=('Absolute_Error','count'),MAE_Cycles=('Absolute_Error','mean'),Bias_Cycles=('Error','mean')).reset_index();metrics.to_csv(OUT/'metrics.csv',index=False)
 (OUT/'manifest.json').write_text(json.dumps(dict(checkpoints=checkpoints,models=MODELS,individual_parameters=True,power_exponent='shared estimated across cells still pre-EOL at checkpoint; intercept and loss coefficient cell-specific',normalization='only for shared power exponent; each model fit per battery',threshold='first integer cycle strictly below 80% of own cycle-2 discharge capacity',input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 print(metrics.to_string(index=False))
if __name__=='__main__':main()
