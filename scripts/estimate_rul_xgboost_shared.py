"""Shared XGBoost trained on both cells' checkpoint-limited histories."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
ROOT=Path(__file__).resolve().parent.parent
INPUT=ROOT/'outputs/capacity_fade/cycle_capacity_analysis.csv'
OUT=ROOT/'outputs/rul_xgboost_shared'
WINDOW=3
HORIZON=1000
SHARED_CHECKPOINTS=[5,8,10,12]
FEATURES=['q_latest','q_previous','q_third','latest_change','mean_two_changes','cycles_since_bol']
PARAMS=dict(n_estimators=100,max_depth=2,learning_rate=.03,min_child_weight=2,reg_lambda=5.,reg_alpha=0.,subsample=1.,colsample_bytree=1.,objective='reg:squarederror',tree_method='hist',random_state=42,n_jobs=1)
def features(q):
 w=np.asarray(q[-3:],dtype=float)
 return [w[-1],w[-2],w[-3],w[-1]-w[-2],(w[-1]-w[0])/2,len(q)-1]
def examples(history):
 q=history.Discharge_Capacity_Ah.to_numpy(float);q=q/q[0];rows=[]
 for j in range(3,len(q)):
  rows.append(dict(zip(FEATURES,features(q[:j])),Target_Change=float(q[j]-q[j-1]),Target_Cycle=int(history.Cycle_Index.iloc[j])))
 return pd.DataFrame(rows,columns=FEATURES+['Target_Change','Target_Cycle'])

def validate_groups(data):
 groups={};eols={}
 for cell,raw in data.groupby('Dataset_ID'):
  g=raw[raw.Computationally_Complete & raw.Discharge_Integration_Valid].sort_values('Cycle_Index').copy()
  if g.Cycle_Index.duplicated().any() or not np.all(np.diff(g.Cycle_Index)==1):raise ValueError(f'{cell}: duplicate/gapped cycles')
  q=g.Discharge_Capacity_Ah.to_numpy(float)
  if not np.isfinite(q).all() or (q<=0).any():raise ValueError(f'{cell}: invalid capacity')
  below=g[g.Discharge_Capacity_Ah<.8*q[0]]
  if below.empty:raise ValueError(f'{cell}: observed EOL required')
  groups[cell]=g;eols[cell]=int(below.Cycle_Index.iloc[0])
 return groups,eols

def forecast(model,history,horizon=HORIZON):
 q=list(history.Discharge_Capacity_Ah.to_numpy(float)/history.Discharge_Capacity_Ah.iloc[0]);start=int(history.Cycle_Index.iloc[-1]);trace=[]
 for step in range(1,horizon+1):
  delta=float(model.predict(pd.DataFrame([features(q)],columns=FEATURES))[0]);v=q[-1]+delta
  if not np.isfinite(v):return None,trace,'nonfinite_prediction'
  q.append(v);trace.append(dict(Forecast_Cycle=start+step,Normalized_Capacity=v,Predicted_Change=delta))
  if v<.8:return start+step,trace,'crossing_predicted'
 return None,trace,'no_crossing_within_horizon'

def main():
 OUT.mkdir(parents=True,exist_ok=True);data=pd.read_csv(INPUT,dtype={'Dataset_ID':str});groups,eols=validate_groups(data)
 results=[];traces=[];train_audits=[];models=[]
 for cp in SHARED_CHECKPOINTS:
  if any(cp>=v for v in eols.values()):continue
  pooled=[]
  for cell,g in groups.items():
   hist=g[g.Cycle_Index<=cp]
   if len(hist)!=cp-1:raise ValueError('Expected contiguous cycle 2 through checkpoint')
   table=examples(hist)
   if table.empty:raise ValueError('No training windows')
   pooled.append(table.assign(Source_Cell=cell))
  train=pd.concat(pooled,ignore_index=True);model=xgb.XGBRegressor(**PARAMS,base_score=float(train.Target_Change.mean()));model.fit(train[FEATURES],train.Target_Change);model.save_model(OUT/f'shared_checkpoint_{cp}.ubj')
  train.to_csv(OUT/f'training_windows_checkpoint_{cp}.csv',index=False)
  for cell,g in groups.items():
   history=g[g.Cycle_Index<=cp];cross,trace,status=forecast(model,history);actual=eols[cell]-cp;pred=cross-cp if cross is not None else np.nan;err=pred-actual
   results.append(dict(Dataset_ID=cell,Checkpoint_Cycle=cp,Training_Windows=len(train),Training_Windows_By_Cell=json.dumps(train.groupby('Source_Cell').size().to_dict()),Actual_EOL_Cycle=eols[cell],Predicted_EOL_Cycle=cross,Actual_RUL_Cycles=actual,Predicted_RUL_Cycles=pred,Signed_Error_Cycles=err,Absolute_Error_Cycles=abs(err),Status=status))
   traces.extend(dict(Dataset_ID=cell,Checkpoint_Cycle=cp,**r,Predicted_Capacity_Ah=r['Normalized_Capacity']*float(g.Discharge_Capacity_Ah.iloc[0])) for r in trace)
  models.append((cp,model))
  fig,axes=plt.subplots(1,2,figsize=(12,5))
  for ax,(cell,g) in zip(axes,groups.items()):
   bol=float(g.Discharge_Capacity_Ah.iloc[0]);past=g[g.Cycle_Index<=cp];later=g[g.Cycle_Index.between(cp,min(int(g.Cycle_Index.max()),eols[cell]+5))]
   ax.plot(past.Cycle_Index,past.Discharge_Capacity_Ah,'o-',color='#146c94',label='Observed through checkpoint',ms=4)
   ax.plot(later.Cycle_Index,later.Discharge_Capacity_Ah,'o--',color='#7a8793',label='Later measured (evaluation only)',ms=4)
   tr=[z for z in traces if z['Dataset_ID']==cell and z['Checkpoint_Cycle']==cp]
   forecast_cycles=[cp]+[z['Forecast_Cycle'] for z in tr];forecast_ah=[float(past.Discharge_Capacity_Ah.iloc[-1])]+[z['Predicted_Capacity_Ah'] for z in tr]
   ax.plot(forecast_cycles,forecast_ah,color='#e07a25',lw=2,label='Shared XGBoost forecast')
   ax.axhline(.8*bol,color='#b02b45',ls=':',lw=2,label=f'EOL 80%: {.8*bol:.3f} Ah');ax.axvline(cp,color='#444',ls=':')
   rr=results[-(2- list(groups).index(cell))] if False else next(z for z in results if z['Dataset_ID']==cell and z['Checkpoint_Cycle']==cp)
   counts=json.loads(rr['Training_Windows_By_Cell']);box=f"Training windows: {rr['Training_Windows']} ({counts[cell]} from this cell + other cell)\nPredicted RUL: {rr['Predicted_RUL_Cycles']} | Actual: {rr['Actual_RUL_Cycles']} cycles"
   ax.text(.97,.96,box,transform=ax.transAxes,ha='right',va='top',fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='#ddd'))
   ax.set(title=f'Battery {cell}',xlabel='Cycle number',ylabel='Discharge capacity (Ah)',xlim=(2,min(55,eols[cell]+5)));ax.grid(alpha=.16)
  h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,fontsize=8);fig.suptitle(f'Shared XGBoost; each battery forecast separately | trained through cycle {cp}');fig.tight_layout(rect=(0,.15,1,.93));fig.savefig(OUT/f'shared_forecast_checkpoint_{cp}.png',dpi=170);plt.close(fig)
 result=pd.DataFrame(results);result.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(traces).to_csv(OUT/'forecast_traces.csv',index=False)
 metrics=result.groupby('Dataset_ID').agg(Checkpoints=('Checkpoint_Cycle','size'),Scored=('Absolute_Error_Cycles','count'),MAE_Cycles=('Absolute_Error_Cycles','mean'),Bias_Cycles=('Signed_Error_Cycles','mean')).reset_index();metrics.to_csv(OUT/'metrics.csv',index=False)
 (OUT/'README.md').write_text('''# Shared XGBoost RUL experiment\n\nOne XGBoost model is trained per checkpoint using causal three-capacity examples from both batteries. Examples are built separately within each cell and then pooled. Each battery is forecast from its own history by recursively predicting normalized next-capacity change until crossing 0.80. EOL is the first observed cycle strictly below 80% of that cell\'s cycle-2 measured capacity. Only checkpoints 5, 8, 10, 12 are used so both cells are pre-EOL. At cycle 8 there are eight pooled examples (four per cell); at cycle 5 only two. Predictions, training examples, model files and Ah-versus-cycle plots are saved here. Future actuals are plotted for evaluation only.\n\nSettings reuse existing exploratory XGBoost parameters. No tuning or early stopping. This is a small two-cell experiment, not an unseen-cell or fleet validation. Loads differ and temperature is unavailable. Failed crossings remain explicitly missing. This script replaces the previous XGBoost experiment in this submission.\n''',encoding='utf8')
 (OUT/'manifest.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoints=[r['Checkpoint_Cycle'] for r in results if r['Dataset_ID']==next(iter(groups))],window=WINDOW,parameters=PARAMS,normalization='each cell by own cycle-2 capacity',forecast='individual recursive forecasts from one shared model',xgboost_version=xgb.__version__),indent=2))
 print(metrics.to_string(index=False));print(result.to_string(index=False))
if __name__=='__main__':main()
