"""Project-specific limitations, risks and the merged-code consistency checklist."""
from pathlib import Path
import html
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,PageBreak,Spacer
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output/pdf/Battery_RUL_Limitations_and_Validation.pdf'
styles=getSampleStyleSheet();styles['Title'].textColor=colors.HexColor('#194b6a');styles['BodyText'].fontSize=9.2;styles['BodyText'].leading=13.5;styles.add(ParagraphStyle(name='GridText',fontSize=8.0,leading=10.7));styles.add(ParagraphStyle(name='LabelHead',fontSize=11,leading=15,fontName='Helvetica-Bold',textColor=colors.HexColor('#17486b'),spaceAfter=6))
def h(s):return Paragraph(s,styles['LabelHead'])
def p(s):return Paragraph(s,styles['BodyText'])
def tab(head,rows,widths=None):
    d=[[Paragraph(html.escape(str(t)),styles['GridText']) for t in head]]+[[Paragraph(html.escape(str(t)),styles['GridText']) for t in r] for r in rows]
    t=Table(d,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f1f5')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d1dce4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),5)]));return t

def main():
 st=[]
 def add(*a):st.extend(a)
 def page(s):
  if st:st.append(PageBreak())
  add(Paragraph(s,styles['Title']),Spacer(1,8))
 page('Battery RUL | Limitations and validation')
 add(p('<b>Basis:</b> two fixed LR1865SZ experimental cell histories, 001_2 and 002_4. This is a scientific limitation report for the updated individual-curve, shared-XGBoost, and minimal-LSTM implementations, not a claim that more data will be collected.'),Spacer(1,10))
 add(h('Seven dataset and interpretation limitations'))
 issues=[
 ('1','Only two batteries','99 usable cycles per cell, not 198 independent battery lifetimes. No unseen-cell generalization can be demonstrated.'),
 ('2','Few early training windows','At cycle 5 each shared model has only two pooled three-cycle supervised windows. ML predictions are fragile and potentially overfit.'),
 ('3','Different current rates','Cells run near 3C and 2C on a 2.4 Ah reference. Rate differences are confounded with differences between individual cells.'),
 ('4','Missing temperature','Cannot estimate measured thermal effects, temperature-dependent degradation parameters, or real pack temperature gradients.'),
 ('5','High-current accessible capacity','Delivered charge under test conditions is not equal to reference-condition SOH; cutoff and polarization may influence decline.'),
 ('6','Relative EOL criterion','EOL uses the first integer cycle below 80% of cycle-2 capacity; it is not manufacturer-rated-capacity EOL.'),
 ('7','No external validation cohort','Chronological holdouts test forecasting of these two histories only, not performance on new batteries or duty cycles.')
 ]
 add(tab(['#','Limitation','Scientific consequence'],issues,[21,143,326]),Spacer(1,12))
 add(p('<b>Defensible project scope:</b> retrospective high-current capacity-fade measurements, exploratory relative-EOL prediction, and transparent two-cell model comparison.'))
 page('Implementation risks and required disclosures')
 add(h('Issues in the replacement models'))
 add(tab(['Issue','Evidence','Proper treatment'],[
 ('XGBoost, cycle 5','Only two pooled training windows; saved model has no feature splits.','Keep as low-data/constant-baseline result, not strong ML evidence.'),
 ('Minimal LSTM stability','At cycle 8, three-cycle LSTM predicts 18 / 4 / 8 for 001_2 and 25 / 9 / 13 for 002_4.','Show individual seeds and spread; avoid confidence-interval claims.'),
 ('Unsuccessful forecasts','The three-cycle LSTM fails to cross EOL for one seed on both cells at cycle 5 (within 1000 steps).','Keep failed rows, report coverage with MAE, do not count failures as zero error.'),
 ('Curve fitting near EOL','Three individual-curve attempts are marked fitted below the EOL threshold at forecast origin.','Exclude from crossing-error MAE but retain as visible failures in attempt tables.'),
 ('Shared power exponent','Later curve checkpoints have just one still-pre-EOL cell.','State that exponent is pooled only when two cells remain eligible.'),
 ('No causality from C-rate','A 3C cell declines faster than a 2C cell, but cells are not matched.','Report association only; no causal C-rate coefficient.'),
 ('Unvalidated pack translation','Synthetic 24-cell envelope assumes temperature and balancing, not observed pack signals.','Call results sensitivity simulations, not predicted pack lifetime.')
 ],[119,184,187]),Spacer(1,8))
 add(p('Forecast target: <b>RUL(c) = observed relative-EOL cycle - checkpoint cycle c</b>. Predictions are the first forecast future integer cycle strictly below the threshold. Full future traces exist for scoring and visual review, not training.'))
 page('All seven RUL methods | common checkpoints')
 add(p('All seven methods are evaluated at checkpoints 8 and 10 for both cells. It retains individual seed attempts. MAE is conditional on successful crossings; coverage is calculated from all attempts.'))
 f=pd.read_csv(ROOT/'outputs/rul_common_checkpoint/all_method_common_metrics.csv',dtype={'Dataset_ID':str})
 rows=[]
 for r in f.itertuples():rows.append([r.Dataset_ID,r.Method.replace(' (individual curve)','').replace('Minimal LSTM (window 3)','LSTM w3').replace('Minimal LSTM (window 1)','LSTM w1'),f'{r.MAE_Cycles:.2f}',f'{r.Scored}/{r.Attempts}',f'{r.Coverage_pct:.0f}%'])
 add(tab(['Cell','Model','MAE (cycles)','Scored','Coverage'],rows,[67,190,87,70,76]),Spacer(1,12))
 add(p('The lower MAE of a simple curve model at these checkpoints is a valid result. It does not imply that XGBoost or LSTM are incorrectly implemented; it shows that their additional flexibility is not reliably useful in this extremely low-data setting.'))
 add(Spacer(1,9),h('Coverage outside the common comparison'))
 a=pd.read_csv(ROOT/'outputs/rul_common_checkpoint/all_checkpoint_metrics.csv',dtype={'Dataset_ID':str})
 for method in ['Minimal LSTM (window 1)','Minimal LSTM (window 3)','Linear (individual curve)','Exponential (individual curve)','Shared XGBoost']:
  sel=a[a.Method==method]
  add(p(f'<b>{html.escape(method)}:</b> '+ '; '.join(f"{r.Dataset_ID}: {int(r.Scored)}/{int(r.Attempts)} scored ({r.Coverage_pct:.1f}%), conditional MAE {r.MAE_Cycles:.2f} cycles" for r in sel.itertuples())))
 add(p('<b>Not an independent battery test:</b> LSTM has three seed runs at each checkpoint, but there remain only two physical batteries.'))
 page('Integration and code-consistency checklist')
 add(h('What was changed in the merged project'))
 add(tab(['Component','Change / validation'],[
 ('RUL scripts','Replaced three original scripts with individual_curve_rul.py, estimate_rul_xgboost_shared.py and estimate_rul_lstm_minimal.py.'),
 ('Prediction artifacts','Removed old three RUL output folders. Added refreshed predictions, metrics, plots, LSTM weights and XGBoost UBJ files.'),
 ('Capacity and pack imports','Updated compare_capacity_models.py and scale_module_pack.py so they do not import deleted RUL functions.'),
 ('Forecast comparison','Regenerated a common-checkpoint table; failures remain visible and model coverage is explicit.'),
 ('Reporting','Regenerated short five-page report, detailed report and Task 6 charts/PDF from new outputs; added complete seven-model / 96-attempt RUL report.'),
 ('Tests','Added merged-model tests for checkpoint causality, error arithmetic, EOL targets, model artifacts, and shared input hashes.'),
 ('Packaging','Rebuilt submission manifest and checked that no old RUL source/output folders or raw Excel files are shipped.')
 ],[140,350]),Spacer(1,12))
 add(h('What is and is not established'))
 add(p('<b>Established by software checks:</b> numerical formulas, checkpoint-limited training examples, saved-output consistency, common input lineage, and runnable model/report scripts.'))
 add(p('<b>Not established:</b> external validity of battery life predictions, chemical SOH, hardware test protocol, calibrated uncertainty, temperature effects, new-cell accuracy, and realistic electrothermal pack performance.'))
 add(p('<b>Reproduction order:</b> capacity-fade source CSV -> the three new RUL scripts -> common-checkpoint evaluation -> Task 6 figures -> PDF reports -> unit tests -> manifest/ZIP. The raw Excel files are omitted; preprocessing requires separate authorized copies.'))
 def footer(c,doc):c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#516575'));c.drawString(50,25,'Two-cell battery analysis | Limitations and code consistency');c.drawRightString(545,25,f'Page {doc.page}')
 OUT.parent.mkdir(parents=True,exist_ok=True)
 SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),leftMargin=49,rightMargin=49,topMargin=40,bottomMargin=50).build(st,onFirstPage=footer,onLaterPages=footer)
 import fitz
 print(OUT,'pages',len(fitz.open(OUT)))
if __name__=='__main__':main()
