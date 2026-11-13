"""Post-review diagnostics; original model selection and outputs remain intact."""
from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fit_capacity_fade import fit,predict,metrics

OUT=ROOT/'outputs/capacity_fade'

def main():
    data=pd.read_csv(OUT/'cycle_capacity_analysis.csv',dtype={'Dataset_ID':str})
    preds=pd.read_csv(OUT/'predictions.csv',dtype={'Dataset_ID':str})
    sensitivity=[]; anchors=[]; bias=[]
    notes=['## Review follow-up: forecast bias and parameter sensitivity','',
        '- These diagnostics were specified after inspecting the existing test results. They do not turn that test window into a fresh independent validation set.',
        '- Original four-panel plots show descriptive full-series fits. New chronological_test_*.png figures show forecasts fitted only through cycle 85.',
        '- Square-root predictions are explicitly conditional on observed future times and are shown separately from cycle-only forecasts.',
        '- An asymptotic exponential is a reasonable future candidate, but it has not been fitted here. Choosing it after seeing these test results would require new rolling-origin validation; its asymptote would not be a guaranteed physical residual capacity.','']
    for ds,g in data.groupby('Dataset_ID'):
        n=g.n.to_numpy(); t=g.t_days.to_numpy(); y=g.Discharge_Capacity_Ah.to_numpy()
        base=fit('Power law',n,t,y); basecurve=predict('Power law',base,n,t)
        masks={'all_cycles':np.ones(len(g),bool),'exclude_cycle_2':g.Cycle_Index.ne(2).to_numpy(),
               'exclude_other_flagged':(~(g.Coulombic_Efficiency_pct.gt(100)&g.Cycle_Index.ne(2))).to_numpy(),
               'exclude_all_flagged':g.Coulombic_Efficiency_pct.le(100).to_numpy()}
        for cycle in g.loc[g.Coulombic_Efficiency_pct.gt(100),'Cycle_Index']:
            masks[f'exclude_only_{int(cycle)}']=g.Cycle_Index.ne(cycle).to_numpy()
        for label,mask in masks.items():
            p=fit('Power law',n[mask],t[mask],y[mask]); curve=predict('Power law',p,n,t); diff=abs(curve-basecurve)
            sensitivity.append(dict(Dataset_ID=ds,Scenario=label,Rows=int(mask.sum()),q0=p[0],a=p[1],b=p[2],
                Max_All_Cycles_Ah=float(diff.max()),Max_Retained_Cycles_Ah=float(diff[mask].max()),
                Max_Change_Cycle=int(g.Cycle_Index.iloc[int(diff.argmax())]),Retained_RMSE_Ah=metrics(y[mask],curve[mask])['RMSE_Ah']))
        q0=y[0]; x=np.sqrt(t); a=max(0.,float(x@(q0-y)/(x@x))); anchored=q0-a*x
        free=fit('Square-root time',n,t,y); freecurve=predict('Square-root time',free,n,t)
        for label,p,curve in [('Free intercept',free,freecurve),('Anchored to measured cycle 2',[q0,a],anchored)]:
            anchors.append(dict(Dataset_ID=ds,Variant=label,q0=p[0],a=p[1],Start_Mismatch_pct=100*(p[0]/q0-1),**metrics(y,curve)))
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        ax.scatter(g.Cycle_Index,y,s=12,color='black',label='Measured')
        ax.plot(g.Cycle_Index,freecurve,label=f'Free intercept: {free[0]:.3f} Ah')
        ax.plot(g.Cycle_Index,anchored,label=f'Anchored: {q0:.3f} Ah')
        ax.set(xlabel='Cycle index',ylabel='Discharge capacity (Ah)',title=f'{ds}: retrospective square-root-time sensitivity')
        ax.legend(); ax.grid(alpha=.2); fig.savefig(OUT/f'anchored_square_root_{ds}.png',dpi=160); plt.close(fig)
        test=preds[(preds.Dataset_ID==ds)&preds.Scope.eq('test_refit')]
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        actual=g[g.Cycle_Index.ge(86)]; persist=float(g.loc[g.Cycle_Index.eq(85),'Discharge_Capacity_Ah'].iloc[0])
        for ax in axes:
            ax.plot(actual.Cycle_Index,actual.Discharge_Capacity_Ah,'ko-',ms=4,label='Measured')
            ax.axhline(persist,color='grey',ls='--',label='Cycle-85 persistence')
            ax.set(xlabel='Cycle index',ylabel='Discharge capacity (Ah)'); ax.grid(alpha=.2)
        for model,rows in test.groupby('Model'):
            target=axes[1] if model=='Square-root time' else axes[0]
            target.plot(rows.Cycle_Index,rows.Predicted_Ah,label=model)
            bias.append(dict(Dataset_ID=ds,Model=model,Mean_Predicted_Minus_Observed_Ah=float((rows.Predicted_Ah-rows.Observed_Ah).mean()),
                Underprediction_Count=int((rows.Predicted_Ah<rows.Observed_Ah).sum()),**metrics(rows.Observed_Ah.to_numpy(),rows.Predicted_Ah.to_numpy())))
        bias.append(dict(Dataset_ID=ds,Model='Persistence',Mean_Predicted_Minus_Observed_Ah=float((persist-actual.Discharge_Capacity_Ah).mean()),Underprediction_Count=int((persist<actual.Discharge_Capacity_Ah).sum()),**metrics(actual.Discharge_Capacity_Ah.to_numpy(),np.full(len(actual),persist))))
        axes[0].set_title(f'{ds}: cycle-only test forecasts')
        axes[1].set_title('Square-root: observed future timing supplied')
        for ax in axes: ax.legend(fontsize=8)
        fig.savefig(OUT/f'chronological_test_{ds}.png',dpi=160); plt.close(fig)
        exp=test[test.Model.eq('Exponential')]; last=exp[exp.Cycle_Index.eq(100)].iloc[0]
        sens=[r for r in sensitivity if r['Dataset_ID']==ds and r['Scenario']=='exclude_all_flagged'][0]
        notes += [f'**{ds}**','',
            f'- Recorded time span: {t[-1]:.4f} days. This is a short cycling experiment; a square-root fit is not evidence of long-term calendar aging.',
            f'- Exponential test mean bias (prediction minus observation): {(exp.Predicted_Ah-exp.Observed_Ah).mean():.6f} Ah. Cycle 100 prediction: {last.Predicted_Ah:.6f} Ah; measured: {last.Observed_Ah:.6f} Ah.',
            f'- Removing all CE-flagged cycles changes the power-law curve by up to {sens["Max_All_Cycles_Ah"]:.6f} Ah across the full original grid; the maximum occurs at cycle {sens["Max_Change_Cycle"]}. Over retained fitting cycles the maximum is {sens["Max_Retained_Cycles_Ah"]:.6f} Ah.',
            f'- Power-law [q0,a,b] changes from {base} to {[sens[k] for k in ["q0","a","b"]]}. The sensitivity is not evidence that the excluded capacities were erroneous.',
            f'- Free square-root intercept: {free[0]:.6f} Ah versus measured {q0:.6f} Ah ({100*(free[0]/q0-1):.2f}% higher). Full-fit RMSE: {metrics(y,freecurve)["RMSE_Ah"]:.6f} Ah free versus {metrics(y,anchored)["RMSE_Ah"]:.6f} Ah anchored.',
            '- Anchoring forces the first point to match but can worsen the rest of the fit. Cycle 2 itself has uncertain SOC conditions; anchoring is a sensitivity exercise, not a correction.', '']
    pd.DataFrame(sensitivity).to_csv(OUT/'power_law_sensitivity_review.csv',index=False)
    pd.DataFrame(anchors).to_csv(OUT/'square_root_anchor_review.csv',index=False)
    pd.DataFrame(bias).to_csv(OUT/'test_bias_comparison.csv',index=False)
    notes+=['**Test comparison (all models retained for transparency)**','','| Cell | Model | RMSE Ah | Mean bias Ah |','|---|---|---:|---:|']
    for r in bias: notes.append(f'| {r["Dataset_ID"]} | {r["Model"]} | {r["RMSE_Ah"]:.6f} | {r["Mean_Predicted_Minus_Observed_Ah"]:.6f} |')
    notes+=['','- Negative bias means systematic underprediction. Negative predictions are still reported rather than clipped.',
        '- Curvature in residuals means the model misses a changing trajectory shape. It does not identify an electrochemical mechanism.',
        '- Parameter sensitivity tables separate removing cycle 2 from removing later flagged cycles. Large changes at the removed origin are extrapolation sensitivity; retained-domain differences must also be examined.',
        '- These are deterministic deletion diagnostics, not statistical confidence intervals. They do not establish population robustness.','']
    text='\n'.join(notes)
    (OUT/'model_review_followup.md').write_text(text,encoding='utf-8')
    report=OUT/'capacity_fade_report.md'; existing=report.read_text(encoding='utf-8').split('## Review follow-up:')[0]
    report.write_text(existing+'\n'+text,encoding='utf-8')
    print(pd.DataFrame(bias).to_string(index=False))
    print(pd.DataFrame(sensitivity).to_string(index=False))

if __name__=='__main__':main()
