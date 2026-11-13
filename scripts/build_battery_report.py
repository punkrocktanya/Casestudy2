"""Regenerate a current detailed technical supplement; no historical RUL files."""
from pathlib import Path
import html
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,PageBreak,Spacer,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'output/pdf/Battery_Tasks_1_to_4_Detailed_Report.pdf'
styles=getSampleStyleSheet();styles['BodyText'].fontSize=9;styles['BodyText'].leading=13;styles['Heading1'].textColor=colors.HexColor('#17486b');styles.add(ParagraphStyle(name='SmallNote',fontSize=8,leading=11,spaceAfter=4))
st=[]
def P(s,style='BodyText'):st.append(Paragraph(str(s),styles[style]));st.append(Spacer(1,7))
def Title(s):P(html.escape(s),'Heading1')
def TableData(frame,fields=None,maxrows=25):
    if fields:frame=frame[fields]
    table_data=[[Paragraph(html.escape(str(v)),styles['SmallNote']) for v in frame.columns]]+[[Paragraph(html.escape(str(v)),styles['SmallNote']) for v in row] for row in frame.head(maxrows).fillna('Not scored').values]
    t=Table(table_data,colWidths=[492/len(frame.columns)]*len(frame.columns),repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1f7')),('GRID',(0,0),(-1,-1),.3,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]));st.append(t);st.append(Spacer(1,10))
def Img(f,width=490,maxh=310):
    p=ROOT/f;w,h=ImageReader(str(p)).getSize();scale=min(width/w,maxh/h);st.append(Image(str(p),width=w*scale,height=h*scale));st.append(Spacer(1,12))

def main():
    fade=pd.read_csv(ROOT/'outputs/capacity_fade/cycle_capacity_analysis.csv',dtype={'Dataset_ID':str})
    curve=pd.read_csv(ROOT/'outputs/rul_individual_curves/predictions.csv',dtype={'Dataset_ID':str})
    lstm=pd.read_csv(ROOT/'outputs/rul_lstm_minimal/predictions.csv',dtype={'Dataset_ID':str})
    xgb=pd.read_csv(ROOT/'outputs/rul_xgboost_shared/predictions.csv',dtype={'Dataset_ID':str})
    fair=pd.read_csv(ROOT/'outputs/rul_common_checkpoint/all_method_common_metrics.csv',dtype={'Dataset_ID':str})
    Title('Battery analysis | Detailed methodology and results')
    P('<b>Updated merged submission - November 2026.</b> This report supersedes the earlier PDF descriptions of the original LSTM, XGBoost and extrapolation experiments. All RUL results below are from the individual-curve, shared-XGBoost and minimal-LSTM implementations in the same ZIP.')
    P('Scope: exactly two LR1865SZ cycling histories, 001_2 (~3C) and 002_4 (~2C) under the experiment convention. Usable capacity targets cover cycles 2 to 100; cycle 1 is not a reference-condition beginning-of-life verification. Raw Excel files are not redistributed.')
    P('<b>Target definition:</b> EOL is first observed integer cycle where delivered discharge capacity is strictly below 0.8 times the cell\'s cycle-2 delivered discharge capacity. This is relative, high-current delivered-capacity EOL, not 80% of nominal-condition manufacturer capacity.')
    summary=[]
    for cell,g in fade.groupby('Dataset_ID'):
        g=g.sort_values('Cycle_Index');q0=float(g.iloc[0].Discharge_Capacity_Ah);ql=float(g.iloc[-1].Discharge_Capacity_Ah)
        summary.append(dict(Cell=cell,Q2_Ah=round(q0,4),Q100_Ah=round(ql,4),Relative_decline_pct=round(100*(1-ql/q0),2),Relative_EOL=int(g[g.Discharge_Capacity_Ah<.8*q0].Cycle_Index.min())))
    TableData(pd.DataFrame(summary))
    Title('Task 1 | Data engineering and quality')
    P('Raw -> normalized clocks -> within-phase interval filtering -> Silver phase reconstruction -> Gold cycle features. Charge / discharge capacities are trapezoidal integrals of current over elapsed test time, divided by 3600 s/h. Energy integrates V * |I|. Time gaps, duplicated samples, unverified phase pairs and CE above 100% remain traceable. Missing temperatures are not imputed.')
    P('Cycle-level capacity for both cells was checked against the instrument-provided capacity counters; agreement supports arithmetic consistency, but the counters may originate from the same current signal and do not independently verify chemical SOH. All Gold records have unverified reference-condition protocol status.')
    Title('Task 2 | Capacity-fade models')
    P('Descriptive models: linear Q=q0-a*n, exponential Q=q0*exp(-k*n), square-root-time, and power law Q=q0-a*n^b. The evaluation distinguishes full-history fit from genuine chronological prediction. It also compares Random Forest, a phenomenological surrogate, persistence and the 18-parameter minimal LSTM in a late-cycle 15-step forecast experiment.')
    Img('outputs/reporting_visualization/01_capacity_predictive_curves.png',maxh=305)
    metrics=pd.read_csv(ROOT/'outputs/capacity_comparison_local/metrics.csv',dtype={'Dataset_ID':str})
    P('<b>Late-cycle test RMSE values (Ah), trained through cycle 85:</b>')
    TableData(metrics[(metrics.Stage=='test')][['Dataset_ID','Model','RMSE_Ah']].round(4),maxrows=10)
    st.append(PageBreak())
    Title('Task 2 | Differential-voltage diagnostics')
    P('dV/dQ is estimated from smoothed voltage-capacity trajectories on representative discharge cycles. The calculation requires consistent support and carefully chosen numerical smoothing. Peak shifts can indicate changes in curve shape, but they do not independently identify lithium inventory loss, loss of active material or irreversible degradation; applied current is high and temperature is missing.')
    Title('Task 3 | Retrospective anomaly screening')
    P('The analysis uses robust residuals against exponential fade, deviations against adjacent cycles and DBSCAN on detrended residuals, relative-capacity-change deviations and CE deviations. Default DBSCAN flags 25 of 98 candidate cycles for 001_2 and 45 of 98 for 002_4. These are review labels, not ground-truth failures. Six CE values above 100% are retained and flagged. Centered features use future cycles and must not be portrayed as online detection.')
    Img('outputs/reporting_visualization/02_coulombic_efficiency.png',maxh=230)
    Img('outputs/reporting_visualization/03_deviation_clusters.png',maxh=280)
    st.append(PageBreak())
    Title('Task 4 | Relative remaining useful life')
    P('Model group A: four individual cell curve fits. Each cell has its own slope/intercept. For power-law, a common exponent is estimated using whichever cells are still before relative EOL at that checkpoint; at checkpoints 15 and 20 this may be just one cell. Forecast starts immediately after the observed checkpoint.')
    P('Model group B: shared XGBoost uses normalized capacity lags, change rates and cycle position. A single model at each checkpoint pools training windows from both cells, then predicts the two cell trajectories separately. At cycle 5 it has only two windows and the fitted trees have no feature splits; it should be treated as a weak constant baseline, not robust learning.')
    P('Model group C: shared minimal LSTM (18 parameters; 1 hidden unit) with one-cycle and three-cycle windows, fixed architecture, 130 epochs and seeds 42, 7 and 123. At checkpoint 5, the three-cycle version has just two pooled supervised windows; seed variability is large and some recursive forecasts never cross the EOL threshold within the 1000-cycle horizon.')
    P('Chronological checkpoints are 5/8/10/12 for shared ML and 5/8/10/12/15/20 for individual curves (the latter two on the still pre-EOL cell). For a fair model comparison, checkpoint 8 and 10 results are compared across both cells; all attempts including failures remain visible.')
    P('<b>Like-for-like MAE and coverage for ALL seven RUL approaches (checkpoints 8 and 10):</b>')
    display=fair.copy()  # all seven RUL methods, not selected four
    display['Method']=display['Method'].str.replace(' (individual curve)','',regex=False)
    TableData(display[['Dataset_ID','Method','MAE_Cycles','Coverage_pct']].round(2))
    Img('outputs/reporting_visualization/04_forecasted_rul_distribution.png',maxh=230)
    P('<b>Complete model record:</b> Battery_RUL_All_Models_Corrected.pdf lists all 96 predictions, including one-cycle/three-cycle LSTM seeds, four empirical curve models, XGBoost, and no-crossing outcomes.')
    st.append(PageBreak())
    Title('Task 4 | Complete result accountability')
    P('The following coverage tables include failed forecasts; MAE is computed on predictions with an EOL crossing and MUST be reported alongside the number scored. Excluded failures are not zero-error forecasts. Seed-to-seed differences reflect initialization sensitivity, not a calibrated predictive interval.')
    for name,filename in [('Individual curves','rul_individual_curves'),('Shared XGBoost','rul_xgboost_shared'),('Minimal LSTM','rul_lstm_minimal')]:
        P(name,'Heading2');m=pd.read_csv(ROOT/'outputs'/filename/'metrics.csv',dtype={'Dataset_ID':str});TableData(m.round(3),maxrows=12)
    Title('Task 4 | Full replacement-model predictions')
    P('The following tables are generated directly from the new RUL prediction CSVs. They are not from the earlier 733-parameter LSTM or the old single-battery XGBoost. Actual RUL is relative to the observed high-current EOL crossing.')
    P('<b>Shared XGBoost: all forecast checkpoints</b>')
    xgb_table=xgb[['Dataset_ID','Checkpoint_Cycle','Actual_RUL_Cycles','Predicted_RUL_Cycles','Signed_Error_Cycles','Training_Windows']].copy()
    TableData(xgb_table,maxrows=8)
    P('<b>Minimal LSTM: both windows, all seeds</b>')
    l=lstm[['Dataset_ID','Window','Checkpoint','Seed','Actual_RUL','Predicted_RUL','Status']].copy()
    TableData(l,maxrows=48)
    P('<b>Individual empirical curves: all 40 attempts</b>')
    c=curve[['Dataset_ID','Model','Checkpoint','Actual_RUL','Predicted_RUL','Status']].copy()
    TableData(c,maxrows=40)
    Title('Task 5 | Illustrative module-to-pack scenarios')
    P('The synthetic 24-cell, 12S2P envelope assumes cell variation, imposed temperature gradients and one-time balancing accessibility. It is not an experimentally fitted electrothermal model, does not solve unequal parallel currents or repeated balancing losses, and cannot validate pack-level lifetime. Simulation outputs are sensitivity demonstrations only.')
    Img('outputs/pack_scaling/scenario_retention.png',maxh=210)
    Title('Reproducibility and boundaries')
    P('Run model scripts in this order: individual_curve_rul.py, estimate_rul_xgboost_shared.py, estimate_rul_lstm_minimal.py, build_common_checkpoint_comparison.py; then build_task6_visualizations.py and both report builders. Requirements, saved trained weights, forecasts, charts, CSV results and manifest are included; the original workbooks are excluded. Physical validation on an unseen battery population is impossible from two histories.')
    def footer(c,d):
        c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#516575'));c.drawString(48,27,'Battery analysis | Updated detailed technical supplement');c.drawRightString(545,27,str(d.page))
    DEST.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(DEST),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=40,bottomMargin=48).build(st,onFirstPage=footer,onLaterPages=footer)
    import fitz
    print(DEST,'pages:',len(fitz.open(DEST)))
if __name__=='__main__':main()
