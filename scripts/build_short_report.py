"""Generate the five-page report from currently saved merged model outputs."""
from pathlib import Path
import html
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'output/pdf/Technical_Report_Module_Outputs.pdf'
BLUE=colors.HexColor('#17486b'); PALE=colors.HexColor('#eaf1f5'); MUTED=colors.HexColor('#516575')
styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='TinyBody',fontName='Helvetica',fontSize=8.4,leading=11.2,spaceAfter=6));styles.add(ParagraphStyle(name='TinyCaption',fontName='Helvetica',fontSize=7.8,leading=10,textColor=MUTED,spaceAfter=5));styles.add(ParagraphStyle(name='SmallHead',fontName='Helvetica-Bold',fontSize=11.5,leading=14,textColor=BLUE,spaceBefore=8,spaceAfter=5)); styles['Title'].textColor=BLUE

def esc(v):return html.escape(str(v)).replace('\n','<br/>')
def p(s,kind='TinyBody'):return Paragraph(s,styles[kind])
def tbl(headers,rows,widths=None):
    grid=[[p(esc(x),'TinyCaption') for x in headers]]+[[p(esc(v),'TinyCaption') for v in row] for row in rows]
    t=Table(grid,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#d7dfe5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    return t

def fig(name,max_height=215,width=490):
    path=ROOT/'outputs/reporting_visualization'/f'{name}.png'
    iw,ih=ImageReader(str(path)).getSize();k=min(width/iw,max_height/ih)
    return Image(str(path),iw*k,ih*k)

def main():
    fade=pd.read_csv(ROOT/'outputs/capacity_fade/cycle_capacity_analysis.csv',dtype={'Dataset_ID':str})
    curve=pd.read_csv(ROOT/'outputs/rul_individual_curves/metrics.csv',dtype={'Dataset_ID':str})
    xgb=pd.read_csv(ROOT/'outputs/rul_xgboost_shared/metrics.csv',dtype={'Dataset_ID':str})
    lstm=pd.read_csv(ROOT/'outputs/rul_lstm_minimal/metrics.csv',dtype={'Dataset_ID':str})
    fair=pd.read_csv(ROOT/'outputs/rul_common_checkpoint/all_method_common_metrics.csv',dtype={'Dataset_ID':str})
    st=[]
    def add(*things):st.extend(things)
    def page(head):
        if st:st.append(PageBreak())
        add(p(esc(head),'Title'),Spacer(1,7))
    page('Battery degradation and RUL | Technical report')
    add(p('Analysis workflow','SmallHead'))
    from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
    flow=Drawing(490,83)
    labels=['1. Source data','2. Bronze','3. Silver','4. Gold','5. Cell analysis','6. RUL models','7. Pack scaling','8. Reporting']
    for i,label in enumerate(labels):
        x=2+(i%4)*124; y=53 if i<4 else 6
        flow.add(Rect(x,y,105,24,fillColor=PALE,strokeColor=colors.HexColor('#a6bdca')))
        flow.add(String(x+52.5,y+8,label,textAnchor='middle',fontName='Helvetica',fontSize=8))
        if i%4<3:
            flow.add(Line(x+106,y+12,x+120,y+12,strokeColor=BLUE))
            flow.add(Polygon([x+120,y+12,x+116,y+15,x+116,y+9],fillColor=BLUE,strokeColor=BLUE))
    flow.add(Line(480,53,480,42,strokeColor=BLUE)); flow.add(Line(480,42,55,42,strokeColor=BLUE)); flow.add(Line(55,42,55,31,strokeColor=BLUE))
    flow.add(Polygon([55,31,52,35,58,35],fillColor=BLUE,strokeColor=BLUE))
    add(flow,Spacer(1,4))
    add(p('<b>Scope.</b> Two LR1865SZ cells, 001_2 and 002_4, with 99 complete discharge observations each (cycles 2-100). The high-current tests measure delivered capacity; polarization and cutoff effects may contribute to the decline. No temperature measurements or verified reference-condition SOH are available.'))
    add(p('Data engineering: medallion structure','SmallHead'))
    add(tbl(['Layer','Main and intermediate steps','Reason'],[
        ['Bronze','Parse timestamps, standardize and validate fields; retain source rows and quality flags. Missing temperature remains missing.','Preserve traceability and avoid inventing measurements.'],
        ['Silver','Classify charge, discharge and rest from current, assign phase IDs and check counters and timing.','Prevent integration across phase boundaries and expose suspect intervals.'],
        ['Gold','Integrate valid within-phase current and voltage intervals; derive cycle capacity, C-rate, efficiency, duration and energy throughput, with quality flags.','Create consistent cycle-level inputs for modelling and anomaly checks.']
    ],[55,244,192]),Spacer(1,5))
    add(p('Repeated elapsed times add zero charge; counter resets and calendar reversals are flagged. C-rate uses a provisional 2.5 Ah divisor. Raw Excel files were not supplied in the package, so raw-to-Gold reproduction requires the original inputs.'))
    add(p('Endpoint and observed results','SmallHead'))
    add(tbl(['Cell','Cycle 2 capacity','80% threshold','First below threshold'],[['001_2','1.5114 Ah','1.2092 Ah','Cycle 13'],['002_4','1.6878 Ah','1.3502 Ah','Cycle 24']],[85,130,130,146]))
    add(p('For exploratory analysis, we assume EOL is the first observed cycle below 80% of each cell’s cycle-2 delivered capacity. If the provisional 2.5 Ah rating is treated as nominal, its 80% threshold is 2.0 Ah and both cells are already below it at their first complete observation. The earlier crossing cannot be dated from these data, and the rating is not verified against a controlled datasheet. The cycle-2 baseline therefore provides an observable relative endpoint, not nominal-capacity SOH.'))

    page('Capacity fade and coulombic efficiency')
    add(fig('01_capacity_predictive_curves',195),p('Figure 1. The dashed line is 80% of cycle-2 capacity; the dotted line and red point identify observed EOL. The plot stops at the first measured crossing. The curves are full-history descriptive fits, not early forecasts.','TinyCaption'))
    add(p('<b>Equations:</b> Linear Q = q0 - a n; exponential Q = q0 exp(-k n); square-root time Q = q0 - a sqrt(t); power law Q = q0 - a n<super>b</super>. Here Q is capacity (Ah), n is cycle index minus 2, t is test-clock days since cycle 2, and q0, a, k and b are fitted parameters.'))
    add(p('These models describe alternative fade shapes; the square-root form is phenomenological. Full-history fits use later data and are not independent predictions.'))
    add(fig('02_coulombic_efficiency',160),p('Figure 2. Coulombic efficiency (%) = 100 x Q_discharge / Q_charge, using integrated capacities (Ah). Median diagnostic efficiencies are 99.11% and 99.26%; six cycles exceed 100% and are flagged for investigation, not clipped or treated as proven faults. Protocol and measurement uncertainty limit interpretation.','TinyCaption'))

    add(tbl(['Cell','Cycles with coulombic efficiency > 100%'],[['001_2','2, 63, 87, 89, 93'],['002_4','2']],[120,371]))
    add(p('Flagged cycles are review candidates; possible measurement, integration, pairing or protocol effects require investigation.','TinyCaption'))

    page('Anomaly detection and degradation signatures')
    add(fig('03_deviation_clusters',245),p('Figure 3. Upper panels project two scaled features used by DBSCAN; lower panels locate flagged cycles on the capacity curves. DBSCAN used residual, capacity-change and efficiency deviations; 25/98 and 45/98 cycles were labelled noise for 001_2 and 002_4.','TinyCaption'))
    add(p('<b>Anomaly process:</b> Fit an exponential capacity trend separately per cell; calculate observed-minus-fit residuals and deviation from the mean of adjacent cycles. Robust scores above 3 flag review; a joint trend candidate requires both tests. For DBSCAN, detrend residual and capacity-change features with an 11-cycle median, add CE deviation, scale by median/IQR, then cluster (eps 0.8; minimum 5). Missing features are excluded.'))
    add(p('<b>Analysis:</b> Cell 001_2 had nine local flags; cell 002_4 had none. Neither had a joint trend candidate. DBSCAN identified 25/98 and 45/98 noise cycles respectively. At eps 0.5 and minimum 5, 001_2 noise rises to 59/98, illustrating parameter sensitivity. Noise means review, not a confirmed fault; centered windows and full-history fits make this retrospective.'))
    add(p('Differential voltage: process and results','SmallHead'))
    add(p('<b>Process:</b> Select discharge cycles 2, 10, 25, 50, 75 and 100. Integrate current over time to obtain capacity Q; consolidate duplicate Q values using median voltage; interpolate voltage onto 301 capacity points. Smooth using three Savitzky-Golay windows (11, 21 and 31 points), calculate dV/dQ, and identify candidate peaks within the interior 10-90% of discharged capacity. Compare candidates across windows and neighbouring cycles.'))
    add(p('<b>Results:</b> Measured discharge capacity decreases from 1.511 to 0.270 Ah for 001_2 and from 1.688 to 0.453 Ah for 002_4 between cycles 2 and 100. The dV/dQ traces also change shape. Of the saved representative candidates, 20/33 for 001_2 and 3/21 for 002_4 pass the window-and-neighbour stability heuristic. These counts depend on detection settings and do not establish matching features or identify electrode mechanisms. Polarization, smoothing and duplicate voltage readings limit interpretation.'))
    paths=[ROOT/'outputs/differential_voltage/revised/interior_comparison_001_2.png',ROOT/'outputs/differential_voltage/revised/interior_comparison_002_4.png']
    imgs=[]
    for path in paths:
        iw,ih=ImageReader(str(path)).getSize(); k=min(238/iw,105/ih); imgs.append(Image(str(path),iw*k,ih*k))
    add(Table([imgs],colWidths=[245,245]),p('Representative dV/dQ interior comparisons. Changes in curves are exploratory loaded-voltage signatures, not proof of lithium or active-material loss.','TinyCaption'))

    page('Early RUL: methods and comparison')
    add(fig('04_forecasted_rul_distribution',125),p('Figure 4. Checkpoint 8: bars show three LSTM seed outcomes; the XGBoost marker and observed RUL provide comparisons. These seed counts are not a population distribution or calibrated confidence interval.','TinyCaption'))
    add(p('At cycle 8, observed RUL was 5 and 16 cycles for 001_2 and 002_4; XGBoost predicted 8 and 14. The three LSTM predictions ranged from 4-18 and 9-25 cycles respectively. The spread shows sensitivity to initialization with very little training data.'))
    add(p('<b>LSTM method:</b> (1) At each checkpoint, retain only observations available through that cycle. (2) Divide each cell’s capacities by its own cycle-2 capacity, setting the exploratory EOL threshold to 0.80. (3) Build one- or three-cycle sequences within each cell and pair them with the next normalized capacity change. (4) Pool the windows and train a shared one-hidden-unit LSTM separately with seeds 42, 7 and 123. (5) Forecast each cell separately: predict the next change, add it to the last capacity and feed the result back until it falls below 0.80. (6) Predicted RUL = predicted EOL cycle minus checkpoint; compare with observed RUL. A non-crossing within the 1,000-cycle horizon is recorded as a failed crossing.'))
    add(p('The histogram uses three-cycle windows at checkpoint 8. Seeds are repeated model runs, not independent batteries. Curve extrapolation uses checkpoint-only data; shared XGBoost learns next-capacity changes from three-capacity windows. Coverage accompanies error because failed crossings are not scored as zero.','TinyCaption'))
    add(p('LSTM forecast curves at checkpoint 8','SmallHead'))
    path=ROOT/'outputs/rul_lstm_minimal/checkpoint_8.png'
    iw,ih=ImageReader(str(path)).getSize(); k=min(490/iw,125/ih); add(Image(str(path),iw*k,ih*k))
    add(p('One- and three-cycle windows are shown; coloured curves represent seeds 42, 7 and 123. Later measurements are for evaluation.','TinyCaption'))
    add(p('XGBoost forecast at checkpoint 8','SmallHead'))
    path=ROOT/'outputs/rul_xgboost_shared/shared_forecast_checkpoint_8.png'
    iw,ih=ImageReader(str(path)).getSize(); k=min(490/iw,115/ih); add(Image(str(path),iw*k,ih*k))
    add(p('The forecast is trained only on histories available through cycle 8; later measurements are shown for evaluation. Only eight pooled training windows are available, and the two cells are not independent external validation.','TinyCaption'))

    page('Cell-to-pack scaling and discussion')
    add(p('Module engineering outputs and limitations','SmallHead'))
    add(tbl(['Output','What it provides','Data needed / status'],[
      ['Capacity and SOH','Cell capacity, retention, mean/range and weakest-cell gap.','Cell values calculated; different test currents confound comparison.'],
      ['Module capacity / energy','Simulated Ah and Wh until the first cell or group reaches a limit.','Needs shared current, initial SOC, voltage model, configuration and verified cutoffs.'],
      ['Voltage / SOC imbalance','Voltage and estimated SOC spread; limiting-cell identity and changes over ageing.','Needs compatible tests and an SOC model; current traces are independent.'],
      ['C-rate and efficiency','Cell C-rate, CE spread and possible charge imbalance.','C-rate divisor provisional; CE alone cannot establish imbalance drift.'],
      ['Temperature and cooling','Cell max/mean, hottest-coldest difference, thermal heatmap and cooling assessment.','Temperatures missing; cooling advice needs layout, limits and validated thermal data/model.'],
      ['Resistance / balancing','Estimated DCIR spread, balancing demand, time and energy loss.','DCIR needs suitable current steps; balancing requires control and hardware parameters.'],
      ['Module RUL / risk','Weakest-cell EOL, module RUL and uncertainty under scenarios.','Requires matched cells and calibrated variability; current estimates are illustrative.'],
      ['Visual outputs','Capacity/SOC/voltage heatmaps, limiting-cell history, thermal maps and operating-window charts.','Use measured values where available; label simulated values and assumptions.']
    ],[120,180,191]),Spacer(1,5))
    add(p('Cell-to-pack extrapolation methodology','SmallHead'))
    add(p('<b>Variation:</b> fit matched-cell fade and voltage models; sample correlated initial capacities and fade rates, group cells by module topology, and simulate the first operating-limit crossing. Repeat to quantify scenario spread. <b>Temperature:</b> use measured cell profiles or a thermal model and independently calibrated temperature-ageing response. <b>Balancing:</b> track cell charge/SOC and apply measured control rules, current limits and losses across cycles.'))
    add(p('<b>Accuracy and scale:</b> simple capacity envelopes support large ensembles cheaply but omit dynamic voltage, current sharing and thermal coupling. Coupled electrothermal/BMS models represent more interactions but require greater data, calibration and computation. Validate a surrogate against detailed simulations before fleet-scale use.'))
    add(p('Data issues and assumptions','SmallHead'))
    add(p('The two tests used different currents and were not simultaneous; timestamp or charge-throughput alignment cannot make them a shared-current series test. Temperature is missing, cutoffs and nominal capacity are provisional, and SOC is not directly available. Voltage at a different current requires a validated model. Missing balancing data do not establish a worst case. Matched cells, verified limits, thermal measurements and balancing behaviour are needed before module predictions can be validated.'))
    add(p('Fleet and battery-passport extensions should retain test conditions, provenance, SOH definition, model version and uncertainty.','TinyCaption'))
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#d2dfe7'));canvas.line(48,40,547,40)
        canvas.setFont('Helvetica',8);canvas.setFillColor(MUTED);canvas.drawString(48,28,'Merged two-cell study | Relative high-current RUL');canvas.drawRightString(547,28,f'{doc.page} / 5')
    DEST.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(DEST),pagesize=(595.28,841.89),leftMargin=48,rightMargin=48,topMargin=40,bottomMargin=51).build(st,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    pages=len(PdfReader(str(DEST)).pages)
    if pages!=5:raise RuntimeError(f'Expected 5 pages; got {pages}')
    print(DEST,'pages:',pages)
if __name__=='__main__':main()

