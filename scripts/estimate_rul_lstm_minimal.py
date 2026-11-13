"""Shared minimal LSTM experiment; preserves existing model outputs."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
INPUT=ROOT/'outputs/capacity_fade/cycle_capacity_analysis.csv'
OUT=ROOT/'outputs/rul_lstm_minimal'
SEEDS=[42,7,123]
SETTINGS=dict(hidden_size=1,layers=1,windows=[3,1],epochs=130,learning_rate=.008,weight_decay=.0001,huber_delta=.02,checkpoints=[5,8,10,12],horizon=1000)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
class MinimalLSTM(nn.Module):
    def __init__(self):
        super().__init__();self.rnn=nn.LSTM(1,1,batch_first=True);self.out=nn.Linear(1,1)
    def forward(self,x):
        z,_=self.rnn(x);return self.out(z[:,-1,:]).squeeze(-1)
def prepare(data,checkpoint,window):
    histories={};xx=[];yy=[];audit=[]
    for cell,g in data[data.Cycle_Index<=checkpoint].groupby('Dataset_ID'):
        g=g[g.Computationally_Complete & g.Discharge_Integration_Valid].sort_values('Cycle_Index')
        c=g.Cycle_Index.to_numpy();q=g.Discharge_Capacity_Ah.to_numpy(float)
        if len(q)<=window or c[0]!=2 or c[-1]!=checkpoint or not np.all(np.diff(c)==1) or not np.isfinite(q).all() or (q<=0).any():raise ValueError('Invalid history')
        q=q/q[0]
        if (q<.8).any():raise ValueError('Observed EOL already reached')
        histories[cell]=q
        for j in range(window,len(q)):
            xx.append(q[j-window:j]);yy.append(q[j]-q[j-1]);audit.append(dict(Dataset_ID=cell,Checkpoint=checkpoint,Window=window,Input_Start_Cycle=int(c[j-window]),Input_End_Cycle=int(c[j-1]),Target_Cycle=int(c[j]),Target_Change=float(q[j]-q[j-1])))
    if len(histories)!=2:raise ValueError('Two cells required')
    return np.array(xx,dtype=np.float32)[...,None],np.array(yy,dtype=np.float32),histories,audit
def train(x,y,seed):
    torch.manual_seed(seed);m=MinimalLSTM();opt=torch.optim.Adam(m.parameters(),lr=SETTINGS['learning_rate'],weight_decay=SETTINGS['weight_decay']);lossfn=nn.HuberLoss(delta=SETTINGS['huber_delta']);losses=[]
    for _ in range(SETTINGS['epochs']):
        opt.zero_grad();loss=lossfn(m(torch.from_numpy(x)),torch.from_numpy(y))
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();opt.step();losses.append(float(loss.detach()))
    m.eval();return m,losses
def recurse(model,q,checkpoint,window,horizon=1000):
    q=list(q);trace=[]
    if any(v<.8 for v in q):return checkpoint,trace,'observed_eol_reached'
    with torch.inference_mode():
        for step in range(1,horizon+1):
            delta=float(model(torch.tensor(q[-window:],dtype=torch.float32).reshape(1,window,1)).item());v=q[-1]+delta
            if not np.isfinite(v):return None,trace,'nonfinite_prediction'
            q.append(v);trace.append(dict(Forecast_Cycle=checkpoint+step,Normalized_Capacity=v,Predicted_Change=delta))
            if v<.8:return checkpoint+step,trace,'crossing_predicted'
    return None,trace,'no_crossing_within_horizon'
def main():
    OUT.mkdir(parents=True,exist_ok=True);data=pd.read_csv(INPUT,dtype={'Dataset_ID':str});groups={c:g[g.Computationally_Complete & g.Discharge_Integration_Valid].sort_values('Cycle_Index') for c,g in data.groupby('Dataset_ID')};eols={c:int(g.loc[g.Discharge_Capacity_Ah<.8*g.Discharge_Capacity_Ah.iloc[0],'Cycle_Index'].min()) for c,g in groups.items()}
    records=[];traces=[];audits=[];losses=[]
    for w in SETTINGS['windows']:
        for cp in SETTINGS['checkpoints']:
            x,y,h,audit=prepare(data,cp,w);audits.extend(audit)
            for seed in SEEDS:
                model,ls=train(x,y,seed);torch.save(model.state_dict(),OUT/f'window{w}_checkpoint{cp}_seed{seed}.pt');losses.extend(dict(Window=w,Checkpoint=cp,Seed=seed,Epoch=i+1,Loss=v) for i,v in enumerate(ls))
                for cell,q in h.items():
                    crossing,trace,status=recurse(model,q,cp,w);rul=crossing-cp if crossing is not None else np.nan;err=rul-(eols[cell]-cp)
                    records.append(dict(Window=w,Checkpoint=cp,Seed=seed,Dataset_ID=cell,Training_Windows=len(y),Predicted_EOL=crossing,Actual_EOL=eols[cell],Predicted_RUL=rul,Actual_RUL=eols[cell]-cp,Error=err,Absolute_Error=abs(err),Status=status));traces.extend(dict(Window=w,Checkpoint=cp,Seed=seed,Dataset_ID=cell,**t) for t in trace)
    r=pd.DataFrame(records);t=pd.DataFrame(traces);r.to_csv(OUT/'predictions.csv',index=False);t.to_csv(OUT/'forecast_traces.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'training_windows.csv',index=False);pd.DataFrame(losses).to_csv(OUT/'training_losses.csv',index=False)
    summary=r.groupby(['Window','Dataset_ID']).agg(Predictions=('Seed','size'),Scored=('Absolute_Error','count'),MAE_Cycles=('Absolute_Error','mean'),Bias_Cycles=('Error','mean')).reset_index();summary.to_csv(OUT/'metrics.csv',index=False)
    for cp in SETTINGS['checkpoints']:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for row,w in enumerate([3,1]):
            for col,(cell,g) in enumerate(groups.items()):
                ax=axes[row,col];bol=float(g.Discharge_Capacity_Ah.iloc[0]);q=g.Discharge_Capacity_Ah;past=g.Cycle_Index<=cp;future=g.Cycle_Index.between(cp,50)
                ax.plot(g.Cycle_Index[past],q[past],'o-',color='#146c94',ms=3,label='Available measurements');ax.plot(g.Cycle_Index[future],q[future],'--',color='#777',label='Later measurements (evaluation)')
                labels=[]
                for seed,color in zip(SEEDS,['#dd7e24','#8a4fa3','#299265']):
                    s=t[(t.Window==w)&(t.Checkpoint==cp)&(t.Dataset_ID==cell)&(t.Seed==seed)];ax.plot([cp]+s.Forecast_Cycle.tolist(),[float(q[g.Cycle_Index==cp].iloc[0])]+(s.Normalized_Capacity*bol).tolist(),color=color,label=f'Seed {seed}')
                    rr=r[(r.Window==w)&(r.Checkpoint==cp)&(r.Dataset_ID==cell)&(r.Seed==seed)].iloc[0];labels.append(str(int(rr.Predicted_RUL)) if pd.notna(rr.Predicted_RUL) else 'missing')
                ax.axhline(.8*bol,color='#b52c45',ls=':',lw=2,label='EOL: 80% of cycle-2 capacity');ax.axvline(cp,color='#aaa',ls=':');ax.set(xlim=(2,50),ylim=(.65*bol,1.08*bol),xlabel='Cycle number',ylabel='Discharge capacity (Ah)',title=f'{cell} | {w}-cycle input | 1 hidden unit');ax.grid(alpha=.15)
                ax.text(.98,.96,'Predicted RUL (seeds 42 / 7 / 123): '+ ' / '.join(labels)+f'\nActual RUL: {eols[cell]-cp} cycles | EOL: {.8*bol:.3f} Ah',transform=ax.transAxes,ha='right',va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='#ddd'))
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9);fig.suptitle(f'Shared minimal LSTM: forecast at cycle {cp} | seed variation is not a confidence interval');fig.tight_layout(rect=(0,.09,1,.95));fig.savefig(OUT/f'checkpoint_{cp}.png',dpi=170);plt.close(fig)
    (OUT/'manifest.json').write_text(json.dumps(dict(settings=SETTINGS,seeds=SEEDS,parameters=sum(p.numel() for p in MinimalLSTM().parameters()),torch_version=torch.__version__,input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),validation='Fixed settings; no tuning or early stopping. No independent validation claimed.',hidden_state='reset per window',normalization='own cycle-2 capacity',forecast='raw recursive change; no forced monotonicity'),indent=2))
    print(summary.to_string(index=False));print(r[r.Checkpoint==8].to_string(index=False))
if __name__=='__main__':main()
