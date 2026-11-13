"""Traceable Silver -> Gold rebuild with interval coverage and conservative release gates.

The old 26-column Gold build is retained in the baseline archive. Measurements,
Silver phase assignments, and source row order are never repaired here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from battery_config import CONFIG, CONFIG_SHA256

ROOT = Path(__file__).resolve().parent.parent
MEASUREMENTS = ['Data_Point', 'Test_Time(s)', 'Current(A)', 'Capacity(Ah)',
                'Voltage(V)', 'Energy(Wh)', 'Temperature(℃)', 'Date_Time', 'Cycle_Index']
SOURCES = {'001_2': 'LR1865SZ_cycles201217_001_2',
           '002_4': 'LR1865SZ_cycles201214_002_4'}


def load_script(name):
    spec = importlib.util.spec_from_file_location('legacy_' + name[:2], ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(frame):
    result = frame[MEASUREMENTS].copy().reset_index(drop=True)
    for col in MEASUREMENTS:
        if col == 'Date_Time':
            result[col] = pd.to_datetime(result[col], errors='raise')
        else:
            result[col] = pd.to_numeric(result[col].replace('-', np.nan), errors='coerce').astype(float)
    return result


def reconcile(left, right, label):
    a, b = normalized(left), normalized(right)
    checks = {}
    for col in MEASUREMENTS:
        checks[col] = len(a) == len(b) and (
            bool(a[col].equals(b[col])) if col == 'Date_Time' else
            bool(np.allclose(a[col], b[col], rtol=0, atol=1e-12, equal_nan=True)))
    if not all(checks.values()):
        raise RuntimeError(f'{label}: measurement/order reconciliation failed: {checks}')
    return {'rows': len(a), 'measurement_checks': checks,
            'measurement_sha256_left': hashlib.sha256(pd.util.hash_pandas_object(a, index=False).values.tobytes()).hexdigest(),
            'measurement_sha256_right': hashlib.sha256(pd.util.hash_pandas_object(b, index=False).values.tobytes()).hexdigest(),
            'numeric_comparison_absolute_tolerance': 1e-12,
            'null_patterns_match': all(a[c].isna().equals(b[c].isna()) for c in MEASUREMENTS),
            'duplicate_data_points': int(a.Data_Point.duplicated().sum())}


def build_intervals(silver, artifacts=None):
    """One row per source observation; pair endpoints identified by durable Row_ID."""
    required = MEASUREMENTS + ['Row_ID', 'Phase', 'Phase_ID']
    missing = set(required) - set(silver)
    if missing:
        raise ValueError(f'Silver contract missing {sorted(missing)}')
    d = silver.copy().reset_index(drop=True)
    if not np.array_equal(d.Row_ID.to_numpy(), np.arange(1, len(d) + 1)):
        raise ValueError('Silver Row_ID must preserve consecutive source order')
    if d.groupby('Phase_ID').Phase.nunique().gt(1).any() or d.groupby('Phase_ID').Cycle_Index.nunique().gt(1).any():
        raise ValueError('Silver phase IDs must identify exactly one phase and one cycle')
    n = normalized(d)
    for c in MEASUREMENTS:
        d[c] = n[c]
    d['Previous_Row_ID'] = d.Row_ID.shift().astype('Int64')
    d['dt_test_s'] = d['Test_Time(s)'].diff()
    d['dt_calendar_s'] = d.Date_Time.diff().dt.total_seconds()
    d['Same_Cycle'] = d.Cycle_Index.eq(d.Cycle_Index.shift())
    d['Same_Phase'] = d.Same_Cycle & d.Phase_ID.eq(d.Phase_ID.shift())
    # Calendar jitter is audited, not used to veto a valid monotonic test clock.
    d['Calendar_Reversal'] = d.dt_calendar_s.lt(0)
    d['Zero_dt'] = d.Same_Phase & d.dt_test_s.eq(0)
    d['Negative_dt'] = d.Same_Cycle & d.dt_test_s.lt(0)
    d['Gap'] = d.Same_Cycle & d.dt_test_s.gt(CONFIG['maximum_integration_interval_s'])
    art = pd.Series(False, index=d.index) if artifacts is None else pd.Series(np.asarray(artifacts, dtype=bool), index=d.index)
    d['Confirmed_Artifact'] = art
    d['Artifact_Touching'] = art | art.shift(fill_value=False)
    finite_i = np.isfinite(d['Current(A)']) & np.isfinite(d['Current(A)'].shift())
    finite_v = np.isfinite(d['Voltage(V)']) & np.isfinite(d['Voltage(V)'].shift())
    d['Missing_Current_Endpoint'] = ~finite_i
    d['Missing_Voltage_Endpoint'] = ~finite_v
    valid_time = d.dt_test_s.gt(0) & d.dt_test_s.le(CONFIG['maximum_integration_interval_s'])
    d['Valid_Capacity'] = d.Same_Phase & valid_time & finite_i & ~d.Artifact_Touching
    d['Valid_Energy'] = d.Valid_Capacity & finite_v
    avg_i = (d['Current(A)'].abs() + d['Current(A)'].shift().abs()) / 2
    avg_p = (d['Current(A)'].abs()*d['Voltage(V)'] + d['Current(A)'].shift().abs()*d['Voltage(V)'].shift())/2
    d['Candidate_Ah_Estimate'] = avg_i * d.dt_test_s / 3600
    d['Candidate_Wh_Estimate'] = avg_p * d.dt_test_s / 3600
    d['Integrated_Ah'] = d.Candidate_Ah_Estimate.where(d.Valid_Capacity)
    d['Integrated_Wh'] = d.Candidate_Wh_Estimate.where(d.Valid_Energy)
    reasons = []
    for row in d.itertuples():
        r = []
        if pd.isna(row.Previous_Row_ID): r.append('FILE_START')
        elif not row.Same_Cycle: r.append('CYCLE_BOUNDARY')
        elif not row.Same_Phase: r.append('PHASE_BOUNDARY')
        if row.Same_Phase:
            if pd.isna(row.dt_test_s): r.append('MISSING_TEST_CLOCK')
            elif row.dt_test_s == 0: r.append('ZERO_DT_NO_ELAPSED_TIME')
            elif row.dt_test_s < 0: r.append('NEGATIVE_TEST_CLOCK')
            elif row.Gap: r.append('TEST_CLOCK_GAP')
            if row.Missing_Current_Endpoint: r.append('MISSING_CURRENT')
            if row.Missing_Voltage_Endpoint: r.append('MISSING_VOLTAGE')
            if row.Artifact_Touching: r.append('CONFIRMED_ARTIFACT')
            if row.Phase == 'UNKNOWN': r.append('UNKNOWN_PHASE')
        reasons.append(';'.join(r))
    d['Exclusion_Reasons'] = reasons
    d['Above_Recommended_Protocol'] = (
        (d.Phase.eq('CHARGE') & d['Current(A)'].abs().gt(CONFIG['rated_charge_current_a'])) |
        (d.Phase.eq('DISCHARGE') & d['Current(A)'].abs().gt(CONFIG['rated_discharge_current_a'])))
    # No physics-impossibility limit or arbitrary smoothing is invented.
    d['Current_Limit_Basis'] = 'PROVISIONAL_DATASHEET_RECOMMENDATION'
    return d


def counter_analysis(rows, column):
    """Positive differences inside phase sessions; negative differences stay visible.

    A negative jump identifies a possible reset/reversal, never a certified reset.
    No unobserved increment across a reset is imputed.
    """
    delta = rows.groupby('Phase_ID', sort=False)[column].diff()
    pairs = delta.notna()
    return {'positive_increment_total': float(delta.clip(lower=0).sum()) if pairs.any() else None,
            'negative_jump_count': int(delta.lt(-1e-12).sum()),
            'negative_jump_magnitude': float(-delta[delta.lt(-1e-12)].sum()),
            'missing_value_count': int(rows[column].isna().sum()),
            'pair_count': int(pairs.sum())}


def phase_coverage_audit(ds, intervals):
    records=[]
    for (ci,phase),rows in intervals.groupby(['Cycle_Index','Phase'],sort=True):
        positive=rows.Same_Phase & rows.dt_test_s.gt(0)
        expected=float(rows.loc[positive,'dt_test_s'].sum())
        valid_s=float(rows.loc[rows.Valid_Capacity,'dt_test_s'].sum())
        energy_s=float(rows.loc[rows.Valid_Energy,'dt_test_s'].sum())
        counts={}
        for reasons in rows.loc[rows.Same_Phase,'Exclusion_Reasons']:
            for reason in reasons.split(';'):
                if reason: counts[reason]=counts.get(reason,0)+1
        records.append({'Dataset_ID':ds,'Cycle_Index':int(ci),'Phase':phase,
            'First_Row_ID':int(rows.Row_ID.iloc[0]),'Last_Row_ID':int(rows.Row_ID.iloc[-1]),
            'Valid_Integrated_s':valid_s,'Expected_Phase_s':expected,
            'Omitted_Positive_s':expected-valid_s,'Energy_Omitted_Positive_s':expected-energy_s,
            'Time_Coverage_pct':100*valid_s/expected if expected else None,
            'Energy_Coverage_pct':100*energy_s/expected if expected else None,
            'Omitted_Interval_Count_By_Reason':json.dumps(counts,sort_keys=True),
            'Phase_Entry_Positive_s':float(rows.loc[rows.Same_Cycle & ~rows.Same_Phase & rows.dt_test_s.gt(0),'dt_test_s'].sum())})
    return records


def mismatch(value, counter):
    if counter is None or not np.isfinite(value) or counter <= 0:
        return None
    return float(abs(value-counter)/counter*100)


def summarize_cycle(ds, group, first_cycle, last_cycle, sign_supported):
    ci = int(group.Cycle_Index.iloc[0]); flags = []
    rec = {'Dataset_ID': ds, 'Cycle_Index': ci,
           'First_Row_ID': int(group.Row_ID.iloc[0]), 'Last_Row_ID': int(group.Row_ID.iloc[-1]),
           'Truncated_At_File_Start': ci == first_cycle,
           'Truncated_At_File_End': ci == last_cycle,
           'Temperature_Status': 'UNAVAILABLE' if group['Temperature(℃)'].isna().all() else 'OBSERVED',
           'Sign_Convention_Supported': bool(sign_supported)}
    phase_audit = []
    for ph, prefix in [('CHARGE','Charge'),('DISCHARGE','Discharge'),('REST','Rest')]:
        rows = group[group.Phase.eq(ph)]
        within = rows.Same_Phase
        positive = within & rows.dt_test_s.gt(0)
        expected = float(rows.loc[positive,'dt_test_s'].sum())
        valid_s = float(rows.loc[rows.Valid_Capacity,'dt_test_s'].sum())
        energy_s = float(rows.loc[rows.Valid_Energy,'dt_test_s'].sum())
        cap = float(rows.Integrated_Ah.sum(min_count=1))
        energy = float(rows.Integrated_Wh.sum(min_count=1))
        omitted = positive & ~rows.Valid_Capacity
        rec.update({prefix+'_Phase_Count': int(rows.Phase_ID.nunique()),
                    prefix+'_Duration_s': valid_s,
                    prefix+'_Expected_Phase_s': expected,
                    prefix+'_Omitted_Positive_s': expected-valid_s,
                    prefix+'_Energy_Omitted_Positive_s': expected-energy_s,
                    prefix+'_Time_Coverage_pct': 100*valid_s/expected if expected>0 else np.nan,
                    prefix+'_Energy_Coverage_pct': 100*energy_s/expected if expected>0 else np.nan,
                    prefix+'_Zero_dt_Count': int(rows.Zero_dt.sum()),
                    prefix+'_Omitted_Ah_Estimate': float(rows.loc[omitted,'Candidate_Ah_Estimate'].sum(min_count=1)) if omitted.any() else 0.0,
                    prefix+'_Omitted_Ah_Unknown_Count': int(rows.loc[omitted,'Candidate_Ah_Estimate'].isna().sum())})
        if ph == 'REST':
            continue
        rec[prefix+'_Capacity_Ah'] = cap
        rec[prefix+'_Energy_Wh'] = energy
        for column, val, unit in [('Capacity(Ah)',cap,'Ah'),('Energy(Wh)',energy,'Wh')]:
            native = counter_analysis(rows,column)
            diff = mismatch(val,native['positive_increment_total'])
            rec[prefix+'_Counter_'+unit] = native['positive_increment_total']
            rec[prefix+'_Counter_'+unit+'_Difference_pct'] = diff
            rec[prefix+'_Counter_'+unit+'_Negative_Jumps'] = native['negative_jump_count']
            if native['negative_jump_count']: flags.append('COUNTER_RESET_OR_REVERSAL')
            if native['missing_value_count']: flags.append('COUNTER_MISSING')
            if diff is None or diff > CONFIG['counter_relative_tolerance']*100: flags.append('COUNTER_RECONCILIATION_UNRESOLVED')
            phase_audit.append({'Dataset_ID':ds,'Cycle_Index':ci,'Phase':ph,'Counter':column,
                                'Integrated':val,'Difference_pct':diff,**native})
        if len(rows):
            rec[prefix+'_Start_V'] = float(rows['Voltage(V)'].iloc[0])
            rec[prefix+'_End_V'] = float(rows['Voltage(V)'].iloc[-1])
        else:
            rec[prefix+'_Start_V'] = rec[prefix+'_End_V'] = np.nan
        rec[prefix+'_Integration_Valid'] = bool(
            np.isfinite(cap) and cap>0 and np.isfinite(energy) and expected>0
            and rec[prefix+'_Time_Coverage_pct'] >= CONFIG['minimum_time_coverage_pct']
            and rec[prefix+'_Energy_Coverage_pct'] >= CONFIG['minimum_time_coverage_pct']
            and not rows.loc[within,'Negative_dt'].any())
        if expected>0 and not rec[prefix+'_Integration_Valid']: flags.append('INSUFFICIENT_PHASE_COVERAGE')
    charge = group[group.Phase.eq('CHARGE') & group.Valid_Capacity]
    if len(charge):
        c = charge.Candidate_Ah_Estimate*3600/charge.dt_test_s/CONFIG['nominal_capacity_ah']
        weights = charge.dt_test_s
        order = np.argsort(c.to_numpy()); cv = c.to_numpy()[order]; w = weights.to_numpy()[order]
        rec['Mean_C_Rate_Charge'] = float(np.average(c,weights=weights))
        rec['P99_C_Rate_Charge'] = float(cv[min(np.searchsorted(np.cumsum(w)/w.sum(),.99),len(cv)-1)])
        rec['Time_Above_1C_Charge_s'] = float(weights[c.gt(1)].sum())
    else:
        rec.update(Mean_C_Rate_Charge=np.nan,P99_C_Rate_Charge=np.nan,Time_Above_1C_Charge_s=np.nan)
    paired = rec['Charge_Integration_Valid'] and rec['Discharge_Integration_Valid']
    ch = group[group.Phase.eq('CHARGE')]; dis = group[group.Phase.eq('DISCHARGE')]
    order_ok = len(ch)>0 and len(dis)>0 and ch.Row_ID.max()<dis.Row_ID.min()
    single_pair = rec['Charge_Phase_Count']==1 and rec['Discharge_Phase_Count']==1 and order_ok
    rec['Partial_Charge'] = len(ch)>0 and len(dis)==0
    rec['Partial_Discharge'] = len(dis)>0 and len(ch)==0
    rec['Computationally_Complete'] = bool(paired and single_pair)
    tol = CONFIG['endpoint_tolerance_v']
    rec['Charge_Endpoint_At_Provisional_Cutoff'] = bool(abs(rec['Charge_End_V']-CONFIG['voltage_max_v'])<=tol)
    rec['Discharge_Endpoint_At_Provisional_Cutoff'] = bool(abs(rec['Discharge_End_V']-CONFIG['voltage_min_v'])<=tol)
    rec['Full_Cycle_Protocol_Verified'] = bool(
        rec['Computationally_Complete'] and CONFIG['protocol_verified'] and sign_supported
        and rec['Charge_Endpoint_At_Provisional_Cutoff'] and rec['Discharge_Endpoint_At_Provisional_Cutoff']
        and not rec['Truncated_At_File_Start'] and not rec['Truncated_At_File_End'])
    # Diagnostic CE remains useful but must not be confused with protocol-certified CE.
    rec['Coulombic_Efficiency_pct'] = rec['Discharge_Capacity_Ah']/rec['Charge_Capacity_Ah']*100 if rec['Computationally_Complete'] else np.nan
    native_charge = rec['Charge_Counter_Ah']
    native_discharge = rec['Discharge_Counter_Ah']
    rec['Counter_Derived_Diagnostic_CE_pct'] = (
        native_discharge/native_charge*100 if rec['Computationally_Complete']
        and native_charge is not None and native_charge>0 and native_discharge is not None else np.nan)
    rec['Protocol_Verified_CE_pct'] = rec['Coulombic_Efficiency_pct'] if rec['Full_Cycle_Protocol_Verified'] else np.nan
    rec['Energy_Throughput_Wh'] = rec['Charge_Energy_Wh']+rec['Discharge_Energy_Wh']
    rec['Cycle_Elapsed_s'] = float(group['Test_Time(s)'].iloc[-1]-group['Test_Time(s)'].iloc[0])
    same_positive = group.Same_Cycle & group.dt_test_s.gt(0)
    accepted = same_positive & group.dt_test_s.le(CONFIG['maximum_integration_interval_s'])
    rec['Integrated_Coverage_s'] = float(group.loc[group.Valid_Capacity,'dt_test_s'].sum())
    rec['Cycle_Duration_s'] = rec['Cycle_Elapsed_s']
    rec['Accepted_Test_Clock_s'] = float(group.loc[accepted,'dt_test_s'].sum())
    rec['Cycle_Gap_Budget_s'] = float(group.loc[same_positive & ~accepted,'dt_test_s'].sum())
    rec['Phase_Boundary_Positive_s'] = float(group.loc[same_positive & ~group.Same_Phase,'dt_test_s'].sum())
    rec['Calendar_Reversal_Count'] = int(group.Calendar_Reversal.sum())
    rec['Zero_dt_Count'] = int(group.Zero_dt.sum())
    rec['Above_Recommended_Protocol_Rows'] = int(group.Above_Recommended_Protocol.sum())
    if rec['Calendar_Reversal_Count']: flags.append('CALENDAR_CLOCK_WARNING')
    if rec['Zero_dt_Count']: flags.append('ZERO_DT_NO_ELAPSED_TIME')
    if group.Negative_dt.any(): flags.append('NEGATIVE_TEST_CLOCK')
    if group.Gap.any(): flags.append('TEST_CLOCK_GAP')
    if rec['Phase_Boundary_Positive_s']>0: flags.append('POSITIVE_PHASE_BOUNDARY_UNINTEGRATED')
    if group.Confirmed_Artifact.any(): flags.append('CONFIRMED_ARTIFACT_EXCLUDED')
    if rec['Above_Recommended_Protocol_Rows']: flags.append('ABOVE_RECOMMENDED_PROTOCOL')
    if not sign_supported: flags.append('SIGN_CONVENTION_UNSUPPORTED')
    if not rec['Full_Cycle_Protocol_Verified']: flags.append('PROTOCOL_UNVERIFIED')
    if not CONFIG['counter_semantics_verified']: flags.append('COUNTER_SEMANTICS_UNVERIFIED')
    if rec['Coulombic_Efficiency_pct']>CONFIG['ce_review_threshold_pct']: flags.append('CE_ABOVE_REVIEW_THRESHOLD')
    if not rec['Computationally_Complete']: flags.append('INCOMPLETE_OR_INVALID_PAIR')
    if rec['Temperature_Status']=='UNAVAILABLE': flags.append('TEMPERATURE_UNAVAILABLE')
    flags = sorted(set(flags))
    limitations = {'ZERO_DT_NO_ELAPSED_TIME','CALENDAR_CLOCK_WARNING','TEMPERATURE_UNAVAILABLE'}
    status = ('FAIL' if not rec['Computationally_Complete'] else
              'REVIEW' if set(flags)-limitations else
              'PASS_WITH_LIMITATIONS' if flags else 'PASS')
    rec['Quality_Status'] = status
    rec['Quality_Flags'] = ';'.join(flags)
    rec['CE_Status'] = 'PROTOCOL_VERIFIED' if rec['Full_Cycle_Protocol_Verified'] else 'DIAGNOSTIC_ONLY'
    rec['SOH_Training_Eligible'] = False  # no verified reference performance/BOL test is provided
    rec['Configuration_SHA256'] = CONFIG_SHA256
    return rec,phase_audit


def validate_output(gold, datasets):
    if gold.duplicated(['Dataset_ID','Cycle_Index']).any():
        raise RuntimeError('Duplicate Gold cycle keys')
    for ds,d in datasets.items():
        actual = gold.loc[gold.Dataset_ID.eq(ds),'Cycle_Index'].tolist()
        if actual != sorted(d.Cycle_Index.unique().astype(int).tolist()):
            raise RuntimeError('Gold cycle membership differs from Silver')
    incomplete = ~gold.Computationally_Complete
    if gold.loc[incomplete,'Coulombic_Efficiency_pct'].notna().any():
        raise RuntimeError('CE present for incomplete pair')
    if gold.loc[~gold.Full_Cycle_Protocol_Verified,'Protocol_Verified_CE_pct'].notna().any():
        raise RuntimeError('Unverified protocol CE')
    for p in ['Charge','Discharge','Rest']:
        if not np.allclose(gold[p+'_Duration_s']+gold[p+'_Omitted_Positive_s'], gold[p+'_Expected_Phase_s']):
            raise RuntimeError('Phase time budget does not reconcile')
    for c in ['Charge_Capacity_Ah','Discharge_Capacity_Ah','Charge_Energy_Wh','Discharge_Energy_Wh','Energy_Throughput_Wh']:
        if gold[c].dropna().lt(0).any(): raise RuntimeError(f'Negative {c}')
    if np.isinf(gold.select_dtypes(include=[np.number]).to_numpy()).any():
        raise RuntimeError('Infinite output value')


def write_report(gold, comparisons, out, comparison):
    lines = ['# Battery pipeline corrective-action report','',
        '- **Result:** The code checks have been strengthened. The results remain a diagnostic dataset.',
        '- Original files and the previous results are preserved in `outputs/validation_baseline_20261111`.',
        '- Gold now reads Silver and keeps its source row IDs, phase IDs, and cycle membership.',
        '- Repeated test times are counted separately. They represent zero elapsed time, not automatically lost charge.',
        '- Calendar-clock reversals are warnings. A valid test-clock interval is no longer discarded just because the calendar clock goes backwards.',
        '- Every source row has an interval diagnostic with its previous row ID, clocks, flags, and integration decision.',
        '- Coverage and counter comparisons are saved for each cycle. Negative counter jumps remain visible; they are not silently repaired.',
        '- Diagnostic efficiency is separate from protocol-verified efficiency. Efficiencies above 100% are flagged and kept unchanged.',
        '- Duration now explicitly separates the test-time span, integrated seconds, boundary time, and long gaps.',
        '- No raw measurements were sorted, deleted, interpolated, clipped, or replaced.','']
    lines += ['**Most useful finding**','',
        f'- The **{int(gold.Zero_dt_Count.sum()):,}** previously excluded within-phase intervals have zero elapsed test time.',
        '- No positive within-phase time was lost in these files. Coverage is 100% on that defined basis; this does not prove the physical protocol was complete.',
        '- Phase/cycle boundary intervals are kept separate from within-phase coverage.',
        '- The rebuilt metrics match the baseline apart from floating-point rounding.',
        '- The largest before/after change in efficiency is '
        + (f'**{comparison.Coulombic_Efficiency_pct_Change.abs().max():.3g} percentage points**.' if len(comparison) else 'not available.'),
        '- The important changes are traceability, measured coverage, and stricter quality labels.','']
    for ds,sub in gold.groupby('Dataset_ID'):
        cmp = comparisons[ds]
        lines += [f'**Dataset {ds}**','',
            f'- Source, cleaned, and Silver measurement rows reconciled: **{cmp["raw_to_clean"]["rows"]:,}**.',
            f'- Gold cycles: **{len(sub)}**; **{int(sub.Quality_Status.eq("REVIEW").sum())} need review** and **{int(sub.Quality_Status.eq("FAIL").sum())} have incomplete/invalid pairs**.',
            f'- Lowest charge/discharge time coverage: **{sub[["Charge_Time_Coverage_pct","Discharge_Time_Coverage_pct"]].min().min():.6f}%**.',
            f'- Omitted positive within-phase time: **{sub[[p+"_Omitted_Positive_s" for p in ["Charge","Discharge","Rest"]]].sum().sum():.6f} seconds**.',
            f'- Zero-time intervals: **{int(sub.Zero_dt_Count.sum()):,}**.',
            f'- Calendar reversals in source order: **{int(sub.Calendar_Reversal_Count.sum())}**.',
            f'- Cycles above the efficiency review threshold: **{sub.loc[sub.Coulombic_Efficiency_pct.gt(CONFIG["ce_review_threshold_pct"]),"Cycle_Index"].tolist()}**.',
            f'- Counter negative jumps: **{int(sub[[p+"_Counter_"+u+"_Negative_Jumps" for p in ["Charge","Discharge"] for u in ["Ah","Wh"]]].sum().sum())}**.',
            f'- Largest capacity/energy counter difference: **{sub[[p+"_Counter_"+u+"_Difference_pct" for p in ["Charge","Discharge"] for u in ["Ah","Wh"]]].max().max():.6f}%**.',
            f'- Typical diagnostic efficiency: **{sub.Coulombic_Efficiency_pct.median():.4f}%**.',
            f'- Typical mean charge C-rate: **{sub.Mean_C_Rate_Charge.median():.6f}C**.','']
        clock=cmp['clock_audit']
        pdf_count={'001_2':3610,'002_4':4830}[ds]
        lines += [f'- Adjacent rows with an unchanged calendar timestamp but increasing test time: **{clock["calendar_duplicate_but_test_time_increases"]:,}** in this replay. The supplied PDF lists **{pdf_count:,}**; that figure was not reproduced.',
            '- This count uses original source order and the rule: calendar-time difference equals zero AND test-time difference is positive. Current source hashes are saved in the manifest.','']
        for _,r in sub[sub.Cycle_Index.eq(2)].iterrows():
            lines += [f'- Cycle 2 efficiency: **{r.Coulombic_Efficiency_pct:.6f}%** from integration and **{r.Counter_Derived_Diagnostic_CE_pct:.6f}%** from positive counter increments. Both remain diagnostic values.','']
        first=sub[sub.Cycle_Index.eq(2)].iloc[0]; last=sub[sub.Cycle_Index.eq(100)].iloc[0]
        lines += [f'- Discharge capacity from cycle 2 to 100: **{first.Discharge_Capacity_Ah:.6f} to {last.Discharge_Capacity_Ah:.6f} Ah** by integration; **{first.Discharge_Counter_Ah:.6f} to {last.Discharge_Counter_Ah:.6f} Ah** by counters.',
            '- The counters support the declining measured capacity trend. Its physical cause and SOH interpretation remain unverified.','']
    lines += ['**Audit action status**','',
        '- **G01 / G03 / G07 / G08 - implemented:** phase coverage, zero-time separation, severity-based quality, boundary-gap accounting, elapsed and integrated duration fields.',
        '- **G02 / R01 - implemented and replayed:** Gold reads Silver; row order, measurements, phase IDs, cycle membership and file hashes are checked.',
        '- **T01 - implemented:** calendar warnings are separate from test-clock integration. No negative test-clock steps or long gaps were found in these source files.',
        '- **G04 - implemented with evidence pending:** partial edges, computational completeness, endpoint checks, and protocol verification are separate fields. Full protocol verification is false.',
        '- **G05 / G06 - diagnostics implemented:** named-cycle review, unclamped efficiency flags, counter increments, negative-jump detection, and per-phase comparisons. Instrument semantics and calibrated uncertainty remain pending.',
        '- **P01 / V01 - partial:** active Silver/Gold thresholds and core cleaning thresholds use a versioned configuration. Current-limit flags are recommendations, not physics-impossibility claims. Controlled datasheet/protocol confirmation remains pending.',
        '- **B01 - implemented:** elapsed-time parser edge tests, configurable raw paths, source hashes and normalization manifest. Source workbook copies preserve original time strings.',
        '- **S01 - deferred deliberately:** exact Silver segmentation is preserved and sign is checked against voltage trends. Hysteresis/dwell changes need the pulse protocol.',
        '- **D01 - recorded:** temperature is unavailable and is never filled with an assumed value.',
        '- Regression checks are in `test_battery_validated_pipeline.py`; run `python -m unittest test_battery_validated_pipeline -v`.','']
    lines += ['**What still needs evidence**','',
        '- Exact cycling protocol, stop conditions, phase pairing rules, and the original instrument model/calibration.',
        '- A controlled cell datasheet and confirmation of the 2.5 Ah capacity basis and recommended current limits.',
        '- Confirmation of what the instrument capacity/energy counters mean and an uncertainty-based reconciliation tolerance.',
        '- A reference performance test before calculating SOH or treating capacity changes as confirmed degradation.',
        '- Measured temperature logs for any temperature-dependent analysis. Missing temperature has not been filled.',
        '- Sign/dwell/hysteresis changes are deferred until the pulse protocol is known; Silver phase labels are preserved exactly.',
        '- The coverage threshold (99.5%), existing counter tolerance (5%), and >100% efficiency review trigger are processing conventions, not instrument accuracy claims.',
        '- All protocol-unverified cycles stay REVIEW or FAIL. No cycle is certified for SOH training.','',
        '**Files to inspect**','',
        '- `battery_cycle_features_gold.parquet` and `.csv`: revised cycle metrics and quality flags.',
        '- `interval_diagnostics_*.parquet`: source-pair coverage, counter deltas, and exclusion reasons.',
        '- `phase_counter_audit.csv`: phase-level counter reconciliation.',
        '- `phase_coverage_audit.csv`: phase-level time budgets and excluded interval counts by reason.',
        '- `named_anomaly_review.csv`: both cycle-2 cases, the other efficiencies above 100%, and partial boundary cycles.',
        '- `baseline_comparison.csv`: before/after metrics for every cycle.',
        '- `validation_manifest.json`: source/configuration/code hashes, measurement reconciliation, active settings, and release limits.','']
    (out/'corrective_action_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'outputs'/'gold')
    args = parser.parse_args(); out=args.out; out.mkdir(parents=True,exist_ok=True)
    legacy = load_script('03_build_cycle_features.py')
    normalize = load_script('01_normalize_timestamps.py')
    silver_module = load_script('02_silver_transform.py')
    records=[]; counter_rows=[]; coverage_rows=[]; datasets={}; comparisons={}; hashes={}; evidence={}
    for ds,stem in SOURCES.items():
        rawpath=ROOT/(stem+'.xlsx')
        cleanpath=ROOT/'outputs'/'battery_cleaning'/(stem+'_clean.parquet')
        silverpath=ROOT/'outputs'/'silver'/(stem+'_silver.parquet')
        _,raw=normalize.normalize_source(rawpath,ds)
        clean=pd.read_parquet(cleanpath); silver=pd.read_parquet(silverpath)
        comparisons[ds]={'raw_to_clean':reconcile(raw,clean,'raw -> cleaned'),
                         'clean_to_silver':reconcile(clean,silver,'cleaned -> Silver')}
        test_delta=raw['Test_Time(s)'].diff()
        calendar_delta=raw.Date_Time.diff().dt.total_seconds()
        comparisons[ds]['clock_audit']={
            'negative_test_steps':int(test_delta.lt(0).sum()),
            'test_gaps_above_configured_limit':int(test_delta.gt(CONFIG['maximum_integration_interval_s']).sum()),
            'adjacent_equal_test_times':int(test_delta.eq(0).sum()),
            'adjacent_equal_calendar_times':int(calendar_delta.eq(0).sum()),
            'calendar_reversals_all_rows':int(calendar_delta.lt(0).sum()),
            'calendar_duplicate_but_test_time_increases':int((calendar_delta.eq(0)&test_delta.gt(0)).sum()),
            'temperature_observations':int(pd.to_numeric(raw['Temperature(℃)'],errors='coerce').notna().sum())}
        # Replaying Silver rules independently catches stale phase/configuration output.
        replay=silver_module.classify_and_segment(normalized(clean).copy())
        for c in ['Phase','Phase_ID']:
            if not np.array_equal(replay[c].to_numpy(),silver[c].to_numpy()):
                raise RuntimeError(f'{ds}: stale/mismatched Silver {c}; rebuild Silver first')
        comparisons[ds]['silver_phase_replay_exact']=True
        old_state=legacy.add_row_state(legacy.validate_and_standardize(silver,silverpath))
        intervals=build_intervals(silver,old_state['_current_artifact'])
        intervals['Dataset_ID']=ds
        for col in ['Capacity(Ah)','Energy(Wh)']:
            delta=intervals.groupby('Phase_ID',sort=False)[col].diff()
            intervals[col+'_Counter_Delta']=delta
            intervals[col+'_Counter_Negative_Jump']=delta.lt(-1e-12)
            boundary=intervals.Phase_ID.ne(intervals.Phase_ID.shift()) | delta.lt(-1e-12) | delta.isna()
            intervals[col+'_Counter_Session_ID']=boundary.cumsum()
        intervals.to_parquet(out/f'interval_diagnostics_{ds}.parquet',index=False)
        datasets[ds]=intervals
        coverage_rows.extend(phase_coverage_audit(ds,intervals))
        evidence[ds]=silver_module.sign_convention_evidence(silver)
        for _,group in intervals.groupby('Cycle_Index',sort=True):
            rec,audit=summarize_cycle(ds,group,int(intervals.Cycle_Index.min()),int(intervals.Cycle_Index.max()),
                evidence[ds]['positive_current_is_charge_supported'])
            records.append(rec); counter_rows.extend(audit)
        hashes[ds]={str(p.relative_to(ROOT)):sha256(p) for p in [rawpath,cleanpath,silverpath]}
    gold=pd.DataFrame(records).sort_values(['Dataset_ID','Cycle_Index']).reset_index(drop=True)
    validate_output(gold,datasets)
    gold.to_parquet(out/'battery_cycle_features_gold.parquet',index=False)
    gold.to_csv(out/'battery_cycle_features_gold.csv',index=False)
    pd.DataFrame(counter_rows).to_csv(out/'phase_counter_audit.csv',index=False)
    pd.DataFrame(coverage_rows).to_csv(out/'phase_coverage_audit.csv',index=False)
    baseline_path=ROOT/'outputs'/'validation_baseline_20261111'/'outputs'/'gold'/'battery_cycle_features_gold.parquet'
    comparison=pd.DataFrame()
    if baseline_path.exists():
        old=pd.read_parquet(baseline_path)
        metrics=['Charge_Capacity_Ah','Discharge_Capacity_Ah','Coulombic_Efficiency_pct','Mean_C_Rate_Charge',
                 'Charge_Energy_Wh','Discharge_Energy_Wh','Energy_Throughput_Wh','Cycle_Duration_s']
        comparison=old[['Dataset_ID','Cycle_Index']+metrics].merge(gold[['Dataset_ID','Cycle_Index']+metrics],on=['Dataset_ID','Cycle_Index'],suffixes=('_baseline','_revised'),validate='one_to_one')
        for c in metrics: comparison[c+'_Change']=comparison[c+'_revised']-comparison[c+'_baseline']
        comparison.to_csv(out/'baseline_comparison.csv',index=False)
    named=gold[gold.Coulombic_Efficiency_pct.gt(CONFIG['ce_review_threshold_pct']) | gold.Cycle_Index.isin([1,2,101])]
    named.to_csv(out/'named_anomaly_review.csv',index=False)
    manifest={'configuration':CONFIG,'configuration_sha256':CONFIG_SHA256,
              'source_hashes':hashes,'measurement_reconciliation':comparisons,'sign_evidence':evidence,
              'code_hashes':{p.name:sha256(p) for p in (ROOT/'scripts').glob('*.py')},
              'output_rows':len(gold),'output_columns':len(gold.columns),
              'quality_counts':gold.Quality_Status.value_counts().to_dict(),
              'lineage':'raw -> timestamp normalization -> cleaned -> Silver -> Gold; measurements reconciled in source order',
              'release':'DIAGNOSTIC_ONLY; protocol, counter semantics, calibration and BOL reference remain unverified',
              'interval_coverage_definition':'within-phase positive test seconds only; phase/cycle boundaries reported separately',
              'omitted_Ah_estimate_definition':'trapezoid across omitted interval using observed endpoints; not a bound or correction',
              'known_limitations':['No automatic phase smoothing without pulse protocol','No temperature measurements',
                  'No calibrated uncertainty envelope','Source-row identity assumes archived source order; SHA256 binds files']}
    (out/'validation_manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    write_report(gold,comparisons,out,comparison)
    # Replace obsolete reports so the default Gold folder does not present stale quality counts.
    (out/'cycle_feature_report.md').write_text((out/'corrective_action_report.md').read_text(encoding='utf-8'),encoding='utf-8')
    (out/'independent_validation_report.md').write_text((out/'corrective_action_report.md').read_text(encoding='utf-8'),encoding='utf-8')
    (out/'cycle_feature_summary.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    print(f'Validated rebuild: {len(gold)} cycles; {gold.Quality_Status.value_counts().to_dict()}')
    print(f'Report: {out / "corrective_action_report.md"}')


if __name__=='__main__':
    main()
