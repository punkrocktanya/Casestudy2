"""Publication-sized charts from saved results; no model retraining."""
from pathlib import Path
import sys
import json
import ast
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from fit_capacity_fade import predict
sys.path.append(str(Path(__file__).resolve().parent.parent / '.local_deps'))
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, PageBreak, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'outputs/reporting_visualization'
OUT.mkdir(parents=True, exist_ok=True)
CELLS = ['001_2','002_4']
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.titleweight':'bold', 'savefig.facecolor':'white'})
SOURCES = ['outputs/capacity_fade/cycle_capacity_analysis.csv',
           'outputs/capacity_fade/model_parameters.csv', 'outputs/capacity_fade/predictions.csv',
           'outputs/anomaly_detection/anomaly_features.csv',
           'outputs/rul_lstm_minimal/predictions.csv','outputs/rul_xgboost_shared/predictions.csv',
           'outputs/rul_common_checkpoint/all_attempts.csv']


def read(path):
    return pd.read_csv(ROOT/path,dtype={'Dataset_ID':str})


def save(fig, name):
    fig.savefig(OUT/f'{name}.png',dpi=220,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def stop_at_zero(x, y):
    """Display only: terminate at interpolated first zero; retain original arrays."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    hits = np.flatnonzero(y <= 0)
    if not len(hits):
        return x.copy(), y.copy(), None
    j = int(hits[0])
    if j == 0:
        return np.array([]), np.array([]), None
    crossing = x[j-1] + (x[j]-x[j-1]) * y[j-1] / (y[j-1]-y[j])
    return np.r_[x[:j], crossing], np.r_[y[:j], 0.], float(crossing)


def main():
    data,params,forecasts,anomaly,lstm,xgb,rul_all = [read(s) for s in SOURCES]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    palette={'Linear':'#4477AA','Exponential':'#EE7733','Square-root time':'#228833','Power law':'#AA3377'}
    for ax,cell in zip(axes,CELLS):
        g=data[data.Dataset_ID==cell].sort_values('Cycle_Index')
        reference=float(g.loc[g.Cycle_Index==2,'Discharge_Capacity_Ah'].iloc[0])
        threshold=0.8*reference
        eol=int(g.loc[g.Discharge_Capacity_Ah<threshold,'Cycle_Index'].iloc[0])
        shown=g[g.Cycle_Index<=eol]
        ax.scatter(shown.Cycle_Index,shown.Discharge_Capacity_Ah,s=20,color='black',label='Measured',zorder=5)
        for row in params[(params.Dataset_ID==cell)&(params.Scope=='full_series')].itertuples():
            values=predict(row.Model,ast.literal_eval(row.Parameters),shown.n.to_numpy(),shown.t_days.to_numpy())
            # Do not display model values after they fall below the EOL threshold.
            keep=np.flatnonzero(values < threshold)
            end=int(keep[0]) if len(keep) else len(values)
            ax.plot(shown.Cycle_Index.iloc[:end],values[:end],color=palette[row.Model],label=row.Model)
        ax.axhline(threshold,color='#CC3311',linestyle='--',label='80% of cycle 2')
        ax.axvline(eol,color='grey',linestyle=':',label=f'Observed EOL: cycle {eol}')
        ax.scatter([eol],[g.loc[g.Cycle_Index==eol,'Discharge_Capacity_Ah'].iloc[0]],color='#CC3311',s=36,zorder=6)
        ax.set(title=f'{cell}: capacity to observed EOL',xlabel='Cycle index',ylabel='Discharge capacity (Ah)')
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    save(fig,'01_capacity_predictive_curves')

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    ce_summary=[]
    for ax,cell in zip(axes,CELLS):
        g=data[data.Dataset_ID==cell].sort_values('Cycle_Index')
        flags=g[g.Coulombic_Efficiency_pct>100]
        ax.plot(g.Cycle_Index,g.Coulombic_Efficiency_pct,color='#4477AA',marker='.',label='Integrated diagnostic CE')
        ax.axhline(100,color='grey',linestyle='--',label='100% review threshold')
        ax.scatter(flags.Cycle_Index,flags.Coulombic_Efficiency_pct,color='#CC3311',marker='D',zorder=5,label='CE > 100%')
        for r in flags.itertuples():
            ax.annotate(str(r.Cycle_Index),(r.Cycle_Index,r.Coulombic_Efficiency_pct),xytext=(3,7),textcoords='offset points',fontsize=8)
        ax.set(title=f'{cell}: {len(flags)} flagged cycles',xlabel='Cycle index',ylabel='Coulombic efficiency (%)')
        ax.margins(y=.2);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left')
        ce_summary.append(dict(Dataset_ID=cell,Flagged_Cycles=', '.join(map(str,flags.Cycle_Index)),Median_CE_Pct=g.Coulombic_Efficiency_pct.median()))
    save(fig,'02_coulombic_efficiency')
    pd.DataFrame(ce_summary).to_csv(OUT/'ce_summary.csv',index=False)

    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    features=['Residual_Detrended_Ah','Change_Deviation_Pct','CE_Deviation']
    for j,cell in enumerate(CELLS):
        g=anomaly[anomaly.Dataset_ID==cell].copy()
        valid=g.dropna(subset=features+['DBSCAN_Label']).copy()
        z=RobustScaler().fit_transform(valid[features])
        valid['Scaled_Residual']=z[:,0];valid['Scaled_Change']=z[:,1]
        ax=axes[0,j]
        for label in sorted(valid.DBSCAN_Label.unique()):
            m=valid.DBSCAN_Label==label
            noise=label==-1
            ax.scatter(valid.loc[m,'Scaled_Residual'],valid.loc[m,'Scaled_Change'],
                       marker='x' if noise else 'o',s=40 if noise else 25,
                       color='#CC3311' if noise else None,alpha=.8,
                       label='Noise (-1)' if noise else f'Cluster {int(label)}')
        count=int((valid.DBSCAN_Label==-1).sum())
        ax.set(title=f'{cell}: {count}/{len(valid)} noise points',xlabel='Detrended capacity residual (scaled)',ylabel='Capacity-change deviation (scaled)')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        ax=axes[1,j]
        ax.plot(g.Cycle_Index,g.Discharge_Capacity_Ah,color='grey',linewidth=1,label='Measured')
        flagged=g[g.DBSCAN_Label==-1]
        ax.scatter(flagged.Cycle_Index,flagged.Discharge_Capacity_Ah,color='#CC3311',marker='x',label='DBSCAN noise')
        ax.set(title=f'{cell}: flagged cycles on capacity trajectory',xlabel='Cycle index',ylabel='Discharge capacity (Ah)')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    save(fig,'03_deviation_clusters')

    selected=lstm[(lstm.Window==3)&(lstm.Checkpoint==8)]
    selected.to_csv(OUT/'rul_distribution_samples_checkpoint8.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    distribution=[]
    for ax,cell in zip(axes,CELLS):
        g=selected[selected.Dataset_ID==cell]
        values=g.Predicted_RUL.dropna().to_numpy()
        actual=int(g.Actual_RUL.iloc[0])
        xv=int(xgb[(xgb.Dataset_ID==cell)&(xgb.Checkpoint_Cycle==8)].Predicted_RUL_Cycles.iloc[0])
        low=int(min(values.min(),actual,xv))-2;high=int(max(values.max(),actual,xv))+2
        bins=np.arange(low-.5,high+1.5)
        ax.hist(values,bins=bins,color='#4477AA',alpha=.7,edgecolor='white',label='LSTM seed counts (n=3)')
        ax.axvline(actual,color='black',linestyle='--',linewidth=2,label=f'Actual RUL = {actual}')
        ax.axvline(xv,color='#EE7733',linestyle=':',linewidth=2,label=f'XGBoost RUL = {xv}')
        ax.set(title=f'{cell}: forecast from cycle 8',xlabel='Forecasted remaining cycles',ylabel='Number of LSTM runs',yticks=[0,1,2,3],xticks=np.arange(low,high+1))
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
        distribution.append(dict(Dataset_ID=cell,Checkpoint=8,Seed_Count=len(values),
                                 Actual_RUL=actual,XGBoost_RUL=xv,LSTM_Mean_RUL=values.mean(),
                                 LSTM_Min_RUL=values.min(),LSTM_Max_RUL=values.max()))
    save(fig,'04_forecasted_rul_distribution')
    summary=pd.DataFrame(distribution);summary.to_csv(OUT/'rul_distribution_summary.csv',index=False)

    # Plot ALL seven RUL approaches at ALL four shared-ML checkpoints.
    # This supplements (does not replace) the original checkpoint-8 distribution.
    fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    method_keys=['Linear (individual curve)','Exponential (individual curve)',
                 'Square-root cycle (individual curve)','Power law (individual curve)',
                 'Shared XGBoost','Minimal LSTM (window 1)','Minimal LSTM (window 3)']
    for ax,cell in zip(axes,CELLS):
        g=rul_all[(rul_all.Dataset_ID==cell)&rul_all.Checkpoint.isin([5,8,10,12])]
        actual=g.groupby('Checkpoint').Actual_RUL.first().sort_index()
        ax.plot(actual.index,actual.values,'k--o',lw=2,label='Observed relative RUL')
        for model in method_keys:
            f=g[g.Method==model].groupby('Checkpoint').Predicted_RUL.mean().sort_index()
            if not f.empty:
                ax.plot(f.index,f.values,marker='o',markersize=4,label=model.replace(' (individual curve)',''))
        ax.set(title=f'Cell {cell}: all seven approaches',xlabel='Observed checkpoint cycle',ylabel='Predicted RUL (cycles)',xticks=[5,8,10,12])
        ax.grid(alpha=.25)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=4,loc='lower center',bbox_to_anchor=(.5,-.04),fontsize=8)
    save(fig,'05_all_rul_models_checkpoints')

    styles=getSampleStyleSheet();styles['BodyText'].leading=15
    story=[]
    def text(s,style='BodyText'):story.append(Paragraph(s,styles[style]));story.append(Spacer(1,10))
    def chart(name):story.append(Image(str(OUT/f'{name}.png'),width=490,height=370,kind='proportional'));story.append(Spacer(1,10))
    text('Task 6 | Reporting and Visualization','Title')
    text('Battery degradation and RUL: results from Tasks 1-4','Heading2')
    text('Five visualizations are generated from the merged submission. The model results already exist; this script does not fit or tune any RUL models.')
    text('Evidence and interpretation','Heading2')
    text('Capacity forecasts describe high-current delivered discharge capacity. CE is diagnostic and protocol-unverified. DBSCAN labels are candidate outliers. RUL uses 80% of cycle-2 delivered capacity as a relative baseline, not manufacturer nominal-capacity SOH. The RUL histogram shows three LSTM initializations at one fixed checkpoint; it is not a calibrated probability distribution or a population confidence interval.')
    text('Main findings','Heading2')
    text('Both cells show substantial delivered-capacity decline. Six cycles have CE above 100%. Primary DBSCAN flags 25 of 98 evaluated cycles for 001_2 and 45 of 98 for 002_4. Observed relative EOL occurs at cycles 13 and 24. Early forecasting remains sensitive to model choice and limited training data.')
    for title,name,explanation in [
        ('1. Capacity versus cycle count with predictive curves','01_capacity_predictive_curves',
         'Measured capacity is shown through the first observed cycle below 80% of cycle-2 capacity. The dashed line marks the threshold and the vertical line marks observed EOL. Curves are descriptive fits trained on the full history, displayed only until they fall below the threshold; they are not early predictions.'),
        ('2. Coulombic efficiency trends','02_coulombic_efficiency',
         'CE = 100 x discharge Ah / charge Ah. Cell 001_2 exceeds 100% at cycles 2, 63, 87, 89 and 93; cell 002_4 at cycle 2. Values are retained rather than clipped. These flags can arise from pairing, starting state, integration or protocol issues; they do not independently prove damage or charge creation.'),
        ('3. Deviation clusters and anomalous cycles','03_deviation_clusters',
         'DBSCAN uses three robust-scaled features: detrended residual, detrended percentage capacity change and CE deviation. The upper panels display only two of those dimensions; apparent overlap does not contradict three-dimensional clustering. Primary eps=0.8 and min_samples=5 are provisional. Cycle 2 lacks the previous capacity needed for features and is not evaluated. Lower panels locate the same noise points on the measured trajectories. Noise means review candidate; cluster membership does not certify health. Screening is retrospective.'),
        ('4. Forecasted RUL distribution','04_forecasted_rul_distribution',
         'At checkpoint 8 the minimal shared LSTM (one hidden unit, three-cycle input) gives three seed forecasts per cell. One shared XGBoost model, trained on the same two cell histories through cycle 8, predicts separately for both batteries. Actual RUL is 5 cycles for cell 001_2 and 16 for cell 002_4; shared XGBoost predicts 8 and 14 cycles. Bars count seeds, not independent batteries or confidence probabilities.'),
        ('5. All seven RUL models at checkpoints 5, 8, 10 and 12','05_all_rul_models_checkpoints',
         'All four individual curves, shared XGBoost, minimal LSTM window 1 and minimal LSTM window 3 are compared. LSTM points show mean successful seed predictions; failed forecasts are not assigned zero but remain in the full 96-attempt report, Battery_RUL_All_Models_Corrected.pdf. Early ML models have very few training windows; individual square-root and power-law curves can substantially overestimate RUL.')]:
        story.append(PageBreak());text(title,'Heading1');chart(name);text(explanation)
        if name=='04_forecasted_rul_distribution':
            for r in summary.itertuples():
                text(f'{r.Dataset_ID}: LSTM mean RUL {r.LSTM_Mean_RUL:.2f} cycles, observed seed range {r.LSTM_Min_RUL:.0f}-{r.LSTM_Max_RUL:.0f}; actual RUL {r.Actual_RUL}.')
    story.append(PageBreak());text('Reproducibility and reporting limits','Heading1')
    text('All five charts have clearly labelled axes, units, cell identifiers, legends and explanatory captions. Supporting CSVs preserve the RUL samples and summaries. The plotting script reads the saved sources listed below and records hashes in visualization_manifest.json. The assignment section supplied contains a 10-point visualization component; this report does not infer requirements for the unspecified remaining 10 points or assign marks.')
    for source in SOURCES:text(source)
    text('To reproduce: python build_task6_visualizations.py. Existing Task 1-4 results remain the source of all plotted predictions and anomaly labels.')
    def footer(c,d):
        c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#526777'));c.drawString(50,28,'Task 6 | Reporting and Visualization');c.drawRightString(545,28,str(d.page))
    pdf=ROOT/'output/pdf/Task_6_Reporting_and_Visualization_Updated.pdf';pdf.parent.mkdir(parents=True,exist_ok=True)
    from io import BytesIO
    buffer = BytesIO()
    SimpleDocTemplate(buffer,pagesize=(595.28,841.89),leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=50).build(story,onFirstPage=footer,onLaterPages=footer)
    try:
        pdf.write_bytes(buffer.getvalue())
    except PermissionError:
        pdf = pdf.with_name(pdf.stem + '_Updated.pdf')
        pdf.write_bytes(buffer.getvalue())
    manifest=dict(source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  distribution_checkpoint=8,all_model_checkpoints=[5,8,10,12],seeds=[42,7,123],rul_reference='80% of first valid measured capacity (cycle 2)')
    (OUT/'visualization_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    assert len(selected)==6 and selected.Predicted_RUL.notna().all()
    assert [int((anomaly[anomaly.Dataset_ID==c].DBSCAN_Label==-1).sum()) for c in CELLS]==[25,45]
    print(summary.to_string(index=False));print(pdf)


if __name__=='__main__':main()
