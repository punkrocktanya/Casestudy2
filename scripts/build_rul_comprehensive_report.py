"""Fully auditable RUL report: all seven approaches, all checkpoints and seed runs.

Source of truth is the merged CSV model outputs, not hard-coded reported metrics.
This script neither trains models nor changes predictions.
"""
from pathlib import Path
import hashlib, html, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader
import fitz

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output/pdf/Battery_RUL_All_Models_Corrected.pdf'
FIG=ROOT/'outputs/rul_all_models_validation'; FIG.mkdir(exist_ok=True,parents=True)
NAVY=colors.HexColor('#163d57'); LIGHT=colors.HexColor('#edf4f8'); GREY=colors.HexColor('#4a6070')
S=getSampleStyleSheet()
S.add(ParagraphStyle(name='ReportTitleNew',fontName='Helvetica-Bold',fontSize=19,leading=23,textColor=NAVY,spaceAfter=10))
S.add(ParagraphStyle(name='ReportHeadingNew',fontName='Helvetica-Bold',fontSize=12,leading=16,textColor=NAVY,spaceBefore=9,spaceAfter=7))
S.add(ParagraphStyle(name='ReportSubNew',fontName='Helvetica-Bold',fontSize=9,leading=12,textColor=NAVY,spaceBefore=8,spaceAfter=5))
S.add(ParagraphStyle(name='ReportBodyNew',fontName='Helvetica',fontSize=8.5,leading=12.2,spaceAfter=7))
S.add(ParagraphStyle(name='ReportCellNew',fontName='Helvetica',fontSize=7.3,leading=9.0))
S.add(ParagraphStyle(name='ReportCaptionNew',fontName='Helvetica',fontSize=7.4,leading=10.2,textColor=GREY,spaceAfter=8))

def p(v,kind='ReportBodyNew'):
    return Paragraph(str(v),S[kind])
def h(v):return p(html.escape(v),'ReportHeadingNew')
def small(v):return p(html.escape(v),'ReportSubNew')
def tab(headers,rows,widths=None,font=7.3):
    data=[[p(html.escape(str(x)),'ReportCellNew') for x in headers]]+[[p(html.escape(str(x)),'ReportCellNew') for x in row] for row in rows]
    obj=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    obj.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),LIGHT),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#ccdce6')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    return obj

def imgs(path,width=495,maxheight=315):
    wi,he=ImageReader(str(path)).getSize();scale=min(width/wi,maxheight/he)
    return Image(str(path),wi*scale,he*scale)

def fmt(v,status=None):
    if pd.isna(v):
        return 'NOT SCORABLE' if status=='fitted_crossing_at_or_before_origin' else 'NO CROSSING'
    return str(int(v))

def main():
    curve=pd.read_csv(ROOT/'outputs/rul_individual_curves/predictions.csv',dtype={'Dataset_ID':str})
    xgb=pd.read_csv(ROOT/'outputs/rul_xgboost_shared/predictions.csv',dtype={'Dataset_ID':str})
    lstm=pd.read_csv(ROOT/'outputs/rul_lstm_minimal/predictions.csv',dtype={'Dataset_ID':str})
    d=pd.read_csv(ROOT/'outputs/rul_common_checkpoint/all_attempts.csv',dtype={'Dataset_ID':str})
    fair=pd.read_csv(ROOT/'outputs/rul_common_checkpoint/all_method_common_metrics.csv',dtype={'Dataset_ID':str})
    allmetrics=pd.read_csv(ROOT/'outputs/rul_common_checkpoint/all_checkpoint_metrics.csv',dtype={'Dataset_ID':str})
    gold=pd.read_csv(ROOT/'outputs/capacity_fade/cycle_capacity_analysis.csv',dtype={'Dataset_ID':str})
    assert len(curve)==40 and len(xgb)==8 and len(lstm)==48 and len(d)==96
    assert len(fair)==14 and set(fair.Method)==set(d.Method)
    assert set(d.Checkpoint)=={5,8,10,12,15,20}
    assert np.allclose(d.Predicted_RUL-d.Actual_RUL,d.Signed_Error,equal_nan=True)
    assert d.Predicted_RUL.isna().sum()==5
    eol={c:int(g[g.Discharge_Capacity_Ah<.8*g.Discharge_Capacity_Ah.iloc[0]].Cycle_Index.min()) for c,g in gold.groupby('Dataset_ID')}
    assert eol=={'001_2':13,'002_4':24}

    # Save explicit audit-ready data and chart source, with no hidden report-specific filtering.
    d.to_csv(FIG/'all_96_attempts.csv',index=False)
    fair.to_csv(FIG/'all_seven_method_common_metrics.csv',index=False)
    cp=[5,8,10,12]
    # Two plots, one per cell, show all four extrapolation curves, XGB, LSTM both window mean;
    # non-crossings are written in tables, not disguised as zeros.
    legend_names={
      'Linear (individual curve)':'Linear', 'Exponential (individual curve)':'Exponential',
      'Square-root cycle (individual curve)':'Square-root','Power law (individual curve)':'Power law',
      'Shared XGBoost':'Shared XGBoost','Minimal LSTM (window 1)':'LSTM w=1 mean',
      'Minimal LSTM (window 3)':'LSTM w=3 mean'}
    for cell in ['001_2','002_4']:
        fig, ax=plt.subplots(figsize=(10.2,4.5))
        ax.plot(cp,[eol[cell]-c for c in cp],'k--o',linewidth=2,markersize=4,label='Observed RUL')
        for method,label in legend_names.items():
            subset=d[(d.Dataset_ID==cell)&(d.Method==method)&(d.Checkpoint.isin(cp))]
            means=subset.groupby('Checkpoint').Predicted_RUL.mean()
            xx=np.array(cp);yy=np.array([means.get(c,np.nan) for c in cp]);
            if np.any(np.isfinite(yy)):
                ax.plot(xx,yy,marker='o',markersize=4,label=label)
        ax.set(title=f'Cell {cell}: RUL predictions at common checkpoint cycles',xlabel='Forecast origin (cycle)',ylabel='Predicted / actual RUL (cycles)',xticks=cp)
        ax.grid(alpha=.25);ax.legend(ncol=4,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.18))
        fig.subplots_adjust(bottom=.30,top=.86)
        fig.savefig(FIG/f'all_models_checkpoint_chart_{cell}.png',dpi=170);plt.close(fig)

    story=[]
    def add(*items): story.extend(items)
    def page(head,subtitle=None):
        if story:story.append(PageBreak())
        add(p(html.escape(head),'ReportTitleNew'))
        if subtitle:add(p(html.escape(subtitle),'ReportCaptionNew'))

    page('RUL forecast audit | ALL new models','Rebuilt from the new CSV model outputs; checkpoints 5/8/10/12 plus later individual-curve checkpoints 15/20. November 2026.')
    add(h('Why this correction was needed'))
    add(p('The earlier five-page and Task 6 PDFs presented only selected relative-RUL results, which made it look as though old model results had been retained. The present report accounts for all 96 new forecast attempts: 40 empirical curves, 8 shared-XGBoost, and 48 minimal-LSTM (one- and three-cycle windows, 3 seeds). No future capacity measurements are used to construct the checkpoint-specific training windows.'))
    add(tab(['Cell','Valid cycle targets','Cycle-2 capacity','Relative EOL','Total observed RUL at checkpoint 5'],[
        [c,99,f"{gold[gold.Dataset_ID==c].Discharge_Capacity_Ah.iloc[0]:.4f} Ah",eol[c],eol[c]-5] for c in ['001_2','002_4']
    ],[60,100,105,90,140]),Spacer(1,10))
    add(h('Exactly what is predicted'))
    add(p('<b>End of life:</b> the first integer test cycle strictly below 80% of that battery\'s cycle-2 measured, high-current delivered discharge capacity. <b>RUL at checkpoint c:</b> EOL cycle minus c. This is a relative experimental target, not manufacturer-reference state of health (SOH).'))
    add(h('Replacement models and evaluation population'))
    add(tab(['Approach','Checkpoints used','Prediction attempts'],[
       ['Individual Linear / Exponential / Square-root / Power law','5, 8, 10, 12 for both cells; 15, 20 for still-pre-EOL cell 002_4','40 (3 unsuccessful)'],
       ['Shared XGBoost, separately forecast per cell','5, 8, 10, 12','8 (all scored)'],
       ['Shared minimal LSTM, window 1','5, 8, 10, 12 x seeds 42/7/123','24 (all scored)'],
       ['Shared minimal LSTM, window 3','5, 8, 10, 12 x seeds 42/7/123','24 (2 no crossings)']
    ],[178,205,112]))
    add(Spacer(1,9),p('Important: LSTM seed runs are multiple numerical initializations of the SAME two cells, not independent battery replicates. MAE excludes unsuccessful crossings and must always be accompanied by coverage.'))

    page('Fair comparison | seven methods','Same physical cells and same future checkpoints 8 and 10; all model families retained.')
    rows=[]
    for r in fair.itertuples():
        rows.append([r.Dataset_ID,legend_names.get(r.Method,r.Method),f'{r.MAE_Cycles:.2f}',f'{r.Bias_Cycles:+.2f}',f'{int(r.Scored)}/{int(r.Attempts)}',f'{r.Coverage_pct:.0f}%'])
    add(tab(['Cell','Model','MAE (cycles)','Bias','Scored / tried','Coverage'],rows,[53,148,93,65,88,48]),Spacer(1,9))
    add(p('MAE here measures error in predicted remaining cycle count to the experimentally defined relative EOL. XGBoost has one prediction per cell/checkpoint; each LSTM variant has three seeds per cell/checkpoint. Therefore, a metric averaged over six LSTM seed forecasts is not directly equivalent in independent sample count to a metric averaged over two deterministic forecasts.'))
    add(h('Unfiltered attempts across all checkpoints'))
    overall=[]
    for r in allmetrics.itertuples():
        overall.append([r.Dataset_ID,legend_names.get(r.Method,r.Method),f'{int(r.Scored)}/{int(r.Attempts)}',f'{r.MAE_Cycles:.2f}',f'{r.Coverage_pct:.1f}%'])
    add(tab(['Cell','Model','Scored/total','Conditional MAE','Coverage'],overall,[60,164,86,111,74]))
    add(Spacer(1,8),p('No method is declared superior solely on these two histories. The square-root and power-law RUL extrapolations have large errors on some checkpoints, while early LSTM and XGBoost can be unstable or weakly identified.'))

    for c in ['001_2','002_4']:
        page(f'Cell {c} | all models over time',f'Observed EOL {eol[c]}; the dashed black line is observed relative RUL, not a fitted model.')
        add(imgs(FIG/f'all_models_checkpoint_chart_{c}.png',maxheight=268),p('LSTM lines use arithmetic means of scored seeds only; at checkpoint 5, the window-3 mean excludes an unscored seed and therefore cannot represent all three attempts. All individual seed results and missing crossings are shown in later tables.','ReportCaptionNew'))
        add(small('Full checkpoint matrix: predicted remaining cycles'))
        rows=[]
        for method in legend_names:
            row=[legend_names[method]]
            for checkpoint in cp:
                subset=d[(d.Dataset_ID==c)&(d.Method==method)&(d.Checkpoint==checkpoint)]
                if len(subset)==1: v=fmt(subset.iloc[0].Predicted_RUL,subset.iloc[0].Status)
                else:
                    vals=subset.sort_values('Seed')
                    v=' / '.join(fmt(r.Predicted_RUL,r.Status) for r in vals.itertuples())
                row.append(v)
            rows.append(row)
        add(tab(['Model', 'Cycle 5','Cycle 8','Cycle 10','Cycle 12'],rows,[143,88,88,88,88]),Spacer(1,7))
        add(p('For LSTM, three values are the forecasts for the three seeds (sorted by seed number). "NO CROSSING" means the 1,000-cycle forecast did not reach EOL; "NOT SCORABLE" means the fitted curve was already below EOL at the checkpoint. Neither means zero error. The observed RUL values at checkpoints 5/8/10/12 are '+', '.join(str(eol[c]-x) for x in cp)+' cycles.'))

    for c in ['001_2','002_4']:
        page(f'Individual curves | cell {c}','Every curve-model prediction at every valid checkpoint; unsuccessful attempts are shown.')
        g=curve[curve.Dataset_ID==c].sort_values(['Checkpoint','Model'])
        rows=[]
        for r in g.itertuples():
            rows.append([str(int(r.Checkpoint)),r.Model,str(int(r.Actual_RUL)),fmt(r.Predicted_RUL,r.Status),('—' if pd.isna(r.Error) else f'{r.Error:+.0f}'),r.Status.replace('_',' ')])
        add(tab(['Checkpoint','Curve','Actual RUL','Predicted RUL','Signed error','Outcome'],rows,[69,130,77,91,73,55]),Spacer(1,9))
        if c=='001_2':
            add(p('Fits use only checkpoint-observed capacity. Later checkpoints 15 and 20 are available only for cell 002_4 because cell 001_2 crossed relative EOL at cycle 13. Invalid already-below-EOL fits remain unscored.'))

    page('Shared XGBoost | all eight forecasts','One shared model per checkpoint; predictions generated separately for each battery.')
    rows=[]
    for r in xgb.sort_values(['Checkpoint_Cycle','Dataset_ID']).itertuples():
        rows.append([str(int(r.Checkpoint_Cycle)),r.Dataset_ID,str(int(r.Training_Windows)),str(int(r.Actual_RUL_Cycles)),fmt(r.Predicted_RUL_Cycles,r.Status),f'{r.Signed_Error_Cycles:+.0f}',r.Status.replace('_',' ')])
    add(tab(['Origin','Cell','Training windows','Actual RUL','Predicted RUL','Error','Outcome'],rows,[48,58,91,74,95,45,84]),Spacer(1,11))
    add(p('<b>Data scarcity:</b> the checkpoint-5 XGBoost model has just two pooled training windows and no tree splits. Its predictions (19 and 23 cycles) should not be described as learned general degradation behavior. Later checkpoint models have more windows but still only two physical cell histories.'))
    add(h('LSTM window 1 and window 3: all 48 saved predictions'))
    add(p('The following page shows every LSTM seed result. Both input-window configurations are included. The seed runs are correlated experiments on the same two histories.'))

    page('Minimal LSTM | every seed and checkpoint','Seed order 7 / 42 / 123. NO CROSSING indicates that the 1,000-cycle forecast did not cross EOL.')
    rows=[]
    for cell in ['001_2','002_4']:
        for win in [1,3]:
            for checkpoint in cp:
                f=lstm[(lstm.Dataset_ID==cell)&(lstm.Window==win)&(lstm.Checkpoint==checkpoint)].set_index('Seed')
                vals=[fmt(f.loc[seed,'Predicted_RUL'],f.loc[seed,'Status']) for seed in [7,42,123]]
                rows.append([cell,str(win),str(checkpoint),str(eol[cell]-checkpoint)]+vals)
    add(tab(['Cell','Window','Origin','Actual','Seed 7','Seed 42','Seed 123'],rows,[65,51,62,59,86,86,86]),Spacer(1,11))
    add(p('At cycle 8 with window 3, predictions are 4/18/8 cycles for cell 001_2 and 9/25/13 for cell 002_4 (seed order 7/42/123), compared with actual RUL 5 and 16. Window-1 and window-3 LSTM results must not be conflated.'))
    add(h('Unsuccessful forecasts and coverage'))
    failures=d[d.Predicted_RUL.isna()]
    rows=[[r.Dataset_ID,legend_names.get(r.Method,r.Method),str(int(r.Checkpoint)),str(r.Seed),r.Status] for r in failures.itertuples()]
    add(tab(['Cell','Model','Origin','Seed','Reason'],rows,[68,155,60,66,146]),Spacer(1,9))

    page('Consistency and what the results establish','This is a report-level reconciliation, not additional battery data or new model training.')
    add(h('Checks against saved CSV sources'))
    checks=[
        ('All forecast attempts accounted for', '40 individual curves + 8 shared XGBoost + 48 minimal LSTM = 96'),
        ('All model families displayed','Linear, Exponential, Square-root-cycle, Power-law, XGBoost, LSTM window 1, LSTM window 3'),
        ('All evaluated checkpoints displayed','ML: 5, 8, 10, 12; curves additionally cell 002_4 at 15 and 20'),
        ('Missing EOL crossings retained','Three invalid curve fits + two LSTM no-crossing forecasts'),
        ('Numerical error arithmetic verified','Predicted RUL minus observed RUL equals recorded signed error'),
        ('Comparability','Same two batteries and forecast checkpoints 8 and 10 for main metrics'),
        ('Source control','Report is generated from saved prediction CSVs; it does not refit model weights')]
    add(tab(['Check','Evidence'],checks,[182,313]),Spacer(1,10))
    add(h('Source fingerprints'))
    src=['outputs/rul_individual_curves/predictions.csv','outputs/rul_xgboost_shared/predictions.csv','outputs/rul_lstm_minimal/predictions.csv','outputs/rul_common_checkpoint/all_method_common_metrics.csv']
    sr=[]
    for rel in src:
        digest=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
        sr.append([rel.split('/')[-2],digest[:20]])
    add(tab(['Source output folder','SHA-256 prefix (first 20 hex chars)'],sr,[235,260]),Spacer(1,10))
    add(p('<b>Scientific boundaries:</b> This study has only two cells, different high C-rates, no measured temperature, and no reference-condition SOH check. Relative EOL is not nominal-capacity EOL. Some prediction errors are large, and numerical repeatability does not establish new-cell generalization.'))
    add(p('<b>Other results in the project:</b> The 15-step late-cycle LSTM *capacity regression* in Task 2 evaluates forecasts in Ah over cycles 86–100; it is a different evaluation target from the early-life relative-RUL LSTM in this report. Both use the minimal model architecture, but their RMSE and RUL MAE cannot be compared as the same metric.'))
    add(p('<b>Reproduction:</b> Run `build_common_checkpoint_comparison.py` and `build_rul_comprehensive_report.py` after the three model generators. The full machine-readable attempts are in `outputs/rul_all_models_validation/all_96_attempts.csv`.'))
    def footer(c,doc):
        c.setFont('Helvetica',8);c.setFillColor(GREY);c.drawString(48,28,'Corrected RUL reporting | Two-cell experimental study')
        c.drawRightString(545,28,f'Page {doc.page}')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),leftMargin=47,rightMargin=47,topMargin=43,bottomMargin=51).build(story,onFirstPage=footer,onLaterPages=footer)
    info={'all_attempts':len(d),'curve_attempts':len(curve),'xgboost_attempts':len(xgb),'lstm_attempts':len(lstm),
          'unscored_attempts':int(d.Predicted_RUL.isna().sum()),'models':len(fair.Method.unique()),'common_rows':len(fair),
          'pages':len(fitz.open(OUT)),
          'source_sha256':{rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in src}}
    (FIG/'report_manifest.json').write_text(json.dumps(info,indent=2))
    print(str(OUT),info)

if __name__=='__main__':main()
