"""Read-only analysis of counters, boundary voltages and capacity decline."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs'/'scientific_review'


def write_report(anomalous,summary):
    lines=['# Review of efficiency, charging current, and capacity decline','',
        '**What was checked**','',
        '- Independently reintegrated current over test time for all 198 computationally complete cycles.',
        '- Independently summed positive capacity-counter increments in each charge/discharge phase. Both methods matched the saved Gold capacities within 1e-12 Ah.',
        '- Examined the rests immediately before charging and after discharging, current at charge termination, and voltage relaxation.',
        '- Raw files, configuration, quality labels and production Parquet outputs were not changed.','',
        '**1. The six efficiencies above 100%**','',
        '| Dataset | Cycle | Integrated charge Ah | Integrated discharge Ah | Counter charge Ah | Counter discharge Ah | Integrated CE % | Counter CE % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in anomalous:
        lines.append(f'| {r["Dataset_ID"]} | {r["Cycle_Index"]} | {r["Charge_Integrated_Ah"]:.6f} | {r["Discharge_Integrated_Ah"]:.6f} | {r["Charge_Counter_Increment_Ah"]:.3f} | {r["Discharge_Counter_Increment_Ah"]:.3f} | {r["Integrated_CE_pct"]:.6f} | {r["Counter_CE_pct"]:.6f} |')
    lines += ['',
        '- The instrument has one capacity column that resets with phases. These are charge/discharge phase counter changes, not separate lifetime cumulative totals.',
        '- Both cycle-2 cases exceed 100% using either method. The discrepancy is therefore not solely created by our integration.',
        '- In the other four cases, the counters give less than 100%. Their integration excess is only 0.004-0.046 mAh.',
        '- Recorded capacity values lie on a 1 mAh grid. That is reporting resolution, not a known instrument accuracy or uncertainty bound.',
        '- The counter and integrated methods use the same experiment and potentially the same sensor calibration. Agreement is a consistency check, not independent calibration.',
        '- No positive within-phase time was omitted for these cycles.','',
        '**Initial and final SOC conditions**','',
        '- Absolute SOC cannot be established: no verified SOC trace, equilibrium OCV-SOC curve, temperature, or reference capacity is available.',
        '- Compare the voltage at the end of the roughly five-minute rest before charging with the rest after discharging:','',
        '| Dataset | Cycle | Before-charge rest V | After-discharge rest V | Discharge minus charge mAh |',
        '|---|---:|---:|---:|---:|']
    for r in anomalous:
        lines.append(f'| {r["Dataset_ID"]} | {r["Cycle_Index"]} | {r["Pre_Charge_Rest_End_V"]:.4f} | {r["Post_Discharge_Rest_End_V"]:.4f} | {-r["Net_Integrated_Charge_mAh"]:.6f} |')
    lines += ['',
        '- For cycle 2, lower final rest voltage is consistent with ending at lower stored charge than the start. This could explain part of the extra discharge, but voltage relaxation, hysteresis and unknown temperature prevent a quantitative SOC claim.',
        '- Reaching 4.2 V under large charging current does not prove 100% SOC. Charging ends near the full applied current: no conventional current taper is recorded in these files.',
        '- Do not clamp efficiencies or assign initial/final SOC percentages from these voltages.','',
        '**2. High charging currents and specification revision**','',
        '- Complete-cycle mean charge current is about 7.20 A in 001_2 and 4.807 A in 002_4. The current remains high at the 4.2 V endpoint.',
        '- The identifiable specification title is LR1865SZ customer specification V0, dated 2014-12-08. Only its document listing/limited preview was verified; it does not prove that revision governed these tested cells. [Specification preview](https://max.book118.com/html/2017/0607/112142634.shtm)',
        '- A published LR1865SZ experiment lists maximum charge current 2.4 A and maximum discharge current 7.2 A. This corroborates the provisional limits but is not the controlled manufacturer specification for these cells. [Primary experimental paper, Table 1](https://doi.org/10.3389/fenrg.2021.773838)',
        '- The original MIT-BIT dataset publisher explicitly describes fixed 1C/2C/3C charging and discharging in a degradation test programme. High-rate operation was intentional in that programme. [Dataset publisher, version 3](https://data.mendeley.com/datasets/kw34hhw7xg/3)',
        '- The local plateaus are consistent with that programme using a 2.4 Ah test basis: about 3C and 2C. Our configured 2.5 Ah reference gives about 2.879C and 1.923C. These are different reference conventions, not a reason to alter measured currents.',
        '- Exact local-file-to-publisher-test mapping and the applicable controlled specification revision remain unconfirmed. Retain above-recommendation flags and keep protocol verification false.','',
        '**3. Severe delivered-capacity decline**','',
        '| Dataset | Discharge Ah, cycle 2 to 100 | Decline | Voltage rebound after discharge, cycle 2 to 100 |',
        '|---|---|---:|---|']
    for ds,s in summary.items():
        a,b=s['cycle_2'],s['cycle_100']
        lines.append(f'| {ds} | {a["Discharge_Integrated_Ah"]:.6f} to {b["Discharge_Integrated_Ah"]:.6f} | {s["delivered_capacity_decline_pct"]:.2f}% | {a["Discharge_Rebound_V"]:.4f} to {b["Discharge_Rebound_V"]:.4f} V |')
    lines += ['',
        '- Native counters show nearly the same declines: 82.21% and 73.15%. This is a real decline in recorded delivered capacity, not a Gold calculation artifact.',
        '- Applied current remains approximately constant within each dataset; the falling capacity is not explained by progressively larger commanded current.',
        '- After discharge stops at about 3.0 V, voltage rebounds substantially over roughly five minutes. The rebound grows with cycle count. Charge-side relaxation also grows.',
        '- This is evidence of increasing non-equilibrium voltage losses/polarization. Such losses can cause earlier voltage cutoff and reduce delivered capacity even if some charge remains accessible at a lower current.',
        '- The charge-rest/discharge-rest voltage window also narrows. Constant-current charging without a recorded CV taper makes full-charge comparability uncertain.',
        '- The available data cannot establish whether irreversible aging or polarization is dominant, or assign percentages to each. Impedance growth can itself be an aging mechanism, so the two are not mutually exclusive.',
        '- Supported conclusion: declining high-rate delivered capacity with increasing polarization indicators; irreversible low-rate capacity loss remains unquantified.','',
        '**What would distinguish the causes**','',
        '- Obtain the exact test schedule, cell batch/specification revision, instrument calibration/counter definitions, and any temperature or reference-performance logs.',
        '- Use matched, controlled-temperature reference-capacity tests at earlier and later life: the same verified full-charge procedure, adequate rest, and the same low discharge current and cutoff.',
        '- Compare low-rate reference-capacity loss with high-rate delivered-capacity loss at the same aging states. Capacity recovered by reducing current/allowing rest supports a rate-dependent contribution; persistent low-rate loss supports irreversible capacity fade.',
        '- Add matched-SOC pulse/impedance measurements to assess resistance growth. The existing five-minute rebound is not a calibrated DCIR measurement.',
        '- These are evidence requirements for resolving the mechanism, not instructions to run new tests outside the cell or laboratory limits.','',
        '**Detailed files**','',
        '- `six_cycle_counter_soc_review.csv`: six-cycle capacities, counter comparisons, net charge, rest voltages and boundaries.',
        '- `cycle_polarization_review.csv`: the same indicators across all 198 complete diagnostic cycles.',
        '- `quantitative_findings.json` and `review_manifest.json`: endpoint summaries, input hashes and checks.','']
    (OUT/'scientific_investigation_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    gold=pd.read_parquet(ROOT/'outputs/gold/battery_cycle_features_gold.parquet')
    anomalous=[]; cycles=[]; summary={}
    for ds in ['001_2','002_4']:
        path=next((ROOT/'outputs/silver').glob('*'+ds+'_silver.parquet'))
        d=pd.read_parquet(path)
        phases=[]
        for pid,p in d.groupby('Phase_ID',sort=False):
            phases.append(dict(pid=int(pid),cycle=int(p.Cycle_Index.iloc[0]),phase=p.Phase.iloc[0],
                row0=int(p.Row_ID.iloc[0]),row1=int(p.Row_ID.iloc[-1]),
                t0=float(p['Test_Time(s)'].iloc[0]),t1=float(p['Test_Time(s)'].iloc[-1]),
                v0=float(p['Voltage(V)'].iloc[0]),v1=float(p['Voltage(V)'].iloc[-1]),
                i0=float(p['Current(A)'].iloc[0]),i1=float(p['Current(A)'].iloc[-1])))
        lookup={x['pid']:i for i,x in enumerate(phases)}
        for _,g in gold[gold.Dataset_ID.eq(ds)&gold.Computationally_Complete].iterrows():
            charge=d[d.Cycle_Index.eq(g.Cycle_Index)&d.Phase.eq('CHARGE')]
            discharge=d[d.Cycle_Index.eq(g.Cycle_Index)&d.Phase.eq('DISCHARGE')]
            for rows,integrated,native in [(charge,g.Charge_Capacity_Ah,g.Charge_Counter_Ah),
                                            (discharge,g.Discharge_Capacity_Ah,g.Discharge_Counter_Ah)]:
                t=rows['Test_Time(s)'].to_numpy(dtype=float)
                current=rows['Current(A)'].abs().to_numpy(dtype=float)
                dt=np.diff(t)
                if not ((dt>=0)&(dt<=5)).all():
                    raise ValueError('Unexpected within-phase test-time interval; review before using simple independent integration')
                independent=float(np.sum((current[1:]+current[:-1])/2*dt)/3600)
                independent_counter=float(rows['Capacity(Ah)'].diff().clip(lower=0).sum())
                if not np.isclose(independent,integrated,rtol=0,atol=1e-12):
                    raise ValueError('Independent integration differs from Gold')
                if not np.isclose(independent_counter,native,rtol=0,atol=1e-12):
                    raise ValueError('Independent counter total differs from Gold')
            cp=int(charge.Phase_ID.iloc[0]); dp=int(discharge.Phase_ID.iloc[0])
            before=phases[lookup[cp]-1] if lookup[cp]>0 and phases[lookup[cp]-1]['phase']=='REST' else None
            after=phases[lookup[dp]+1] if lookup[dp]+1<len(phases) and phases[lookup[dp]+1]['phase']=='REST' else None
            mid=phases[lookup[cp]+1] if phases[lookup[cp]+1]['phase']=='REST' else None
            r={'Dataset_ID':ds,'Cycle_Index':int(g.Cycle_Index),
               'Charge_Integrated_Ah':g.Charge_Capacity_Ah,'Discharge_Integrated_Ah':g.Discharge_Capacity_Ah,
               'Charge_Counter_Increment_Ah':g.Charge_Counter_Ah,'Discharge_Counter_Increment_Ah':g.Discharge_Counter_Ah,
               'Integrated_CE_pct':g.Coulombic_Efficiency_pct,'Counter_CE_pct':g.Counter_Derived_Diagnostic_CE_pct,
               'Charge_Integration_Minus_Counter_mAh':1000*(g.Charge_Capacity_Ah-g.Charge_Counter_Ah),
               'Discharge_Integration_Minus_Counter_mAh':1000*(g.Discharge_Capacity_Ah-g.Discharge_Counter_Ah),
               'Net_Integrated_Charge_mAh':1000*(g.Charge_Capacity_Ah-g.Discharge_Capacity_Ah),
               'Charge_End_Current_A':float(charge['Current(A)'].iloc[-1]),
               'Discharge_Mean_Current_A':g.Discharge_Capacity_Ah*3600/g.Discharge_Duration_s,
               'Charge_Mean_Current_A':g.Charge_Capacity_Ah*3600/g.Charge_Duration_s,
               'Charge_End_V':g.Charge_End_V,'Discharge_End_V':g.Discharge_End_V,
               'Charge_Duration_s':g.Charge_Duration_s,'Discharge_Duration_s':g.Discharge_Duration_s,
               'Pre_Charge_Rest_End_V':before['v1'] if before else np.nan,
               'Post_Discharge_Rest_End_V':after['v1'] if after else np.nan,
               'Pre_Charge_Rest_s':before['t1']-before['t0'] if before else np.nan,
               'Post_Discharge_Rest_s':after['t1']-after['t0'] if after else np.nan,
               'Post_Charge_Rest_End_V':mid['v1'] if mid else np.nan,
               'Discharge_Rebound_V':after['v1']-g.Discharge_End_V if after else np.nan,
               'Charge_Relaxation_V':g.Charge_End_V-mid['v1'] if mid else np.nan,
               'Start_End_Rest_V_Difference':after['v1']-before['v1'] if after and before else np.nan,
               'Charge_Omitted_Positive_s':g.Charge_Omitted_Positive_s,
               'Discharge_Omitted_Positive_s':g.Discharge_Omitted_Positive_s,
               'Absolute_SOC_Available':False}
            cycles.append(r)
            if g.Coulombic_Efficiency_pct>100: anomalous.append(r)
        s=pd.DataFrame([r for r in cycles if r['Dataset_ID']==ds])
        first=s[s.Cycle_Index.eq(2)].iloc[0]; last=s[s.Cycle_Index.eq(100)].iloc[0]
        summary[ds]={'cycle_2':first.to_dict(),'cycle_100':last.to_dict(),
            'delivered_capacity_decline_pct':float(100*(1-last.Discharge_Integrated_Ah/first.Discharge_Integrated_Ah)),
            'native_counter_decline_pct':float(100*(1-last.Discharge_Counter_Increment_Ah/first.Discharge_Counter_Increment_Ah)),
            'mean_charge_current_range_A':[float(s.Charge_Mean_Current_A.min()),float(s.Charge_Mean_Current_A.max())],
            'mean_discharge_current_range_A':[float(s.Discharge_Mean_Current_A.min()),float(s.Discharge_Mean_Current_A.max())],
            'charge_terminal_current_range_A':[float(s.Charge_End_Current_A.min()),float(s.Charge_End_Current_A.max())],
            'source_temperature_observed_rows':int(d['Temperature(℃)'].notna().sum())}
    pd.DataFrame(anomalous).to_csv(OUT/'six_cycle_counter_soc_review.csv',index=False)
    pd.DataFrame(cycles).to_csv(OUT/'cycle_polarization_review.csv',index=False)
    (OUT/'quantitative_findings.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    write_report(anomalous,summary)
    inputs=list((ROOT/'outputs/silver').glob('*silver.parquet'))+[ROOT/'outputs/gold/battery_cycle_features_gold.parquet',Path(__file__)]
    (OUT/'review_manifest.json').write_text(json.dumps({
        'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'checks':'198 complete diagnostic cycles; phase capacities independently reintegrated and positive instrument counter increments independently summed; all matched Gold to 1e-12 Ah',
        'scope':'No source, configuration or production Gold file modified'},indent=2),encoding='utf-8')
    print(pd.DataFrame(anomalous)[['Dataset_ID','Cycle_Index','Charge_Integrated_Ah','Discharge_Integrated_Ah','Charge_Counter_Increment_Ah','Discharge_Counter_Increment_Ah','Integrated_CE_pct','Counter_CE_pct','Net_Integrated_Charge_mAh','Pre_Charge_Rest_End_V','Post_Discharge_Rest_End_V']].to_string(index=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
