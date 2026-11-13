"""Normalize heterogeneous RUL outputs and compare common, predeclared checkpoints.

Primary fair comparison uses checkpoint cycles 8 and 10 for both cells, four
model families (Linear, Exponential, Shared XGBoost, Minimal LSTM window=3).
Every attempted prediction is retained, including no-crossing/invalid cases.
"""
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/rul_common_checkpoint'
COMMON=(8,10)

def normalized_predictions():
    curve=pd.read_csv(ROOT/'outputs/rul_individual_curves/predictions.csv',dtype={'Dataset_ID':str})
    xgb=pd.read_csv(ROOT/'outputs/rul_xgboost_shared/predictions.csv',dtype={'Dataset_ID':str})
    lstm=pd.read_csv(ROOT/'outputs/rul_lstm_minimal/predictions.csv',dtype={'Dataset_ID':str})
    rows=[]
    for r in curve.itertuples():
        rows.append(dict(Dataset_ID=r.Dataset_ID,Method=r.Model+' (individual curve)',Checkpoint=int(r.Checkpoint),Seed='deterministic',Actual_RUL=float(r.Actual_RUL),Predicted_RUL=r.Predicted_RUL,Status=r.Status))
    for r in xgb.itertuples():
        rows.append(dict(Dataset_ID=r.Dataset_ID,Method='Shared XGBoost',Checkpoint=int(r.Checkpoint_Cycle),Seed='deterministic',Actual_RUL=float(r.Actual_RUL_Cycles),Predicted_RUL=r.Predicted_RUL_Cycles,Status=r.Status))
    for r in lstm.itertuples():
        rows.append(dict(Dataset_ID=r.Dataset_ID,Method='Minimal LSTM (window '+str(int(r.Window))+')',Checkpoint=int(r.Checkpoint),Seed=str(int(r.Seed)),Actual_RUL=float(r.Actual_RUL),Predicted_RUL=r.Predicted_RUL,Status=r.Status))
    d=pd.DataFrame(rows)
    d['Signed_Error']=d.Predicted_RUL-d.Actual_RUL
    d['Absolute_Error']=d.Signed_Error.abs()
    d['Scored']=d.Status.eq('crossing_predicted') & d.Predicted_RUL.notna()
    return d

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=normalized_predictions()
    d.to_csv(OUT/'all_attempts.csv',index=False)
    metrics=(d.groupby(['Dataset_ID','Method'],dropna=False).agg(Attempts=('Scored','size'),Scored=('Scored','sum'),MAE_Cycles=('Absolute_Error','mean'),Bias_Cycles=('Signed_Error','mean')).reset_index())
    metrics['Coverage_pct']=(100*metrics.Scored/metrics.Attempts).round(2)
    metrics.to_csv(OUT/'all_checkpoint_metrics.csv',index=False)
    methods=['Linear (individual curve)','Exponential (individual curve)','Shared XGBoost','Minimal LSTM (window 3)']
    common=d[d.Method.isin(methods) & d.Checkpoint.isin(COMMON)].copy()
    assert set(common.Dataset_ID)=={'001_2','002_4'}
    assert len(common)==(2*len(COMMON)*(1+1+1+3))
    # Include unscored cases, rather than silently intersecting surviving predictions.
    common.to_csv(OUT/'predictions.csv',index=False)
    fair=common.groupby(['Dataset_ID','Method']).agg(Attempts=('Scored','size'),Scored=('Scored','sum'),MAE_Cycles=('Absolute_Error','mean'),Bias_Cycles=('Signed_Error','mean')).reset_index()
    fair['Coverage_pct']=100*fair.Scored/fair.Attempts
    fair.to_csv(OUT/'metrics.csv',index=False)
    # Inclusive, same-checkpoint evaluation for ALL seven RUL approaches.
    # Keep original four-family comparison files for backwards compatibility.
    all_common=d[d.Checkpoint.isin(COMMON)].copy()
    all_common.to_csv(OUT/'all_method_common_predictions.csv',index=False)
    all_fair=all_common.groupby(['Dataset_ID','Method']).agg(
        Attempts=('Scored','size'), Scored=('Scored','sum'),
        MAE_Cycles=('Absolute_Error','mean'), Bias_Cycles=('Signed_Error','mean')).reset_index()
    all_fair['Coverage_pct']=100*all_fair.Scored/all_fair.Attempts
    all_fair.to_csv(OUT/'all_method_common_metrics.csv',index=False)
    (OUT/'README.md').write_text('''# Common-checkpoint evaluation\n\n`all_attempts.csv` retains every attempted forecast from individual curves, shared XGBoost and minimal LSTM (both windows, all three seeds). `all_checkpoint_metrics.csv` reports MAE conditional on successful EOL crossing AND coverage.\n\nThe primary four-family comparison (`predictions.csv`, `metrics.csv`) predefines checkpoint cycles **8 and 10** for both cells. It compares individual Linear, individual Exponential, shared XGBoost and shared minimal three-cycle LSTM (three seeds). All attempts are retained, including failures. Each seed is a distinct initialization, **not** an independent battery. The common evaluation has two distinct cells, and is not external validation.\n\n`all_method_common_predictions.csv` and `all_method_common_metrics.csv` extend the SAME checkpoints (8 and 10) to all four empirical curve models, shared XGBoost and both minimal-LSTM window lengths, with no model exclusions.\n\nEOL is first integer cycle below 80% of *own cycle-2 high-current delivered capacity*, not nominal-condition SOH.\n''',encoding='utf8')
    print(fair.to_string(index=False))
if __name__=='__main__': main()
