#!/usr/bin/env python3
"""Entry point for the validated Silver -> Gold build.

The functions below preserve the original diagnostic algorithm for baseline
comparison and artifact detection. Direct execution runs battery_validated_pipeline,
which exports expanded coverage/counter diagnostics and conservative quality gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "battery_thresholds.json"
DEFAULT_INPUTS = {
    "001_2": ROOT / "outputs" / "battery_cleaning" / "LR1865SZ_cycles201217_001_2_clean.parquet",
    "002_4": ROOT / "outputs" / "battery_cleaning" / "LR1865SZ_cycles201214_002_4_clean.parquet",
}
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gold"
REQUIRED_COLUMNS = [
    "Data_Point", "Test_Time(s)", "Current(A)", "Capacity(Ah)",
    "Voltage(V)", "Energy(Wh)", "Temperature(℃)", "Date_Time", "Cycle_Index",
]
GOLD_COLUMNS = [
    "Dataset_ID", "Cycle_Index", "Cycle_Status", "Cycle_Complete",
    "Charge_Phase_Count", "Discharge_Phase_Count", "Rest_Phase_Count",
    "Mean_C_Rate_Charge", "P99_C_Rate_Charge", "Time_Above_1C_Charge_s",
    "Charge_Capacity_Ah", "Discharge_Capacity_Ah", "Coulombic_Efficiency_pct",
    "Charge_Duration_s", "Discharge_Duration_s", "Rest_Duration_s", "Cycle_Duration_s",
    "Charge_Energy_Wh", "Discharge_Energy_Wh", "Energy_Throughput_Wh",
    "Charge_Integration_Valid", "Discharge_Integration_Valid",
    "Timestamp_Issue_Count", "Invalid_Interval_Count", "Quality_Flag_Count", "Quality_Status",
]

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing threshold configuration: {CONFIG_PATH}")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
NOMINAL_CAPACITY_AH = float(CONFIG["nominal_capacity_ah"])
REST_THRESHOLD_A = float(CONFIG["rest_current_threshold_a"])
MAX_INTERVAL_SECONDS = float(CONFIG["maximum_integration_interval_s"])
RATED_CHARGE_CURRENT_A = float(CONFIG["rated_charge_current_a"])
HAMPEL_WINDOW = int(CONFIG["hampel_window_samples"])
HAMPEL_SIGMA = float(CONFIG["hampel_sigma"])
MINIMUM_PERSISTENT_SAMPLES = int(CONFIG["minimum_persistent_samples"])
CURRENT_JUMP_THRESHOLD_A = float(CONFIG["current_jump_threshold_a"])
NEIGHBOUR_TOLERANCE_A = float(CONFIG["neighbour_agreement_tolerance_a"])
COUNTER_RELATIVE_TOLERANCE = float(CONFIG["counter_relative_tolerance"])


def validate_and_standardize(raw: pd.DataFrame, source: Path) -> pd.DataFrame:
    duplicates = raw.columns[raw.columns.duplicated()].tolist()
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if duplicates or missing:
        raise RuntimeError(f"{source.name}: schema failure; missing={missing}, duplicates={duplicates}")
    if raw.empty:
        raise RuntimeError(f"{source.name}: empty input")
    df = raw.loc[:, REQUIRED_COLUMNS].copy()
    for column in [c for c in REQUIRED_COLUMNS if c != "Date_Time"]:
        original = df[column]
        sentinels = ["", "-", "--", "NA", "N/A", "null", "None"]
        if column == "Temperature(℃)":
            sentinels += ["-999", "65535", "32767"]
        cleaned = original.astype("string").str.strip().replace(sentinels, pd.NA)
        converted = pd.to_numeric(cleaned, errors="coerce")
        failures = original.notna() & converted.isna()
        if failures.any() and column != "Temperature(℃)":
            raise RuntimeError(f"{source.name}: unparseable {column} values")
        df[column] = converted
    for column in ["Data_Point", "Cycle_Index"]:
        fractional = df[column].notna() & df[column].mod(1).ne(0)
        if fractional.any():
            raise RuntimeError(f"{source.name}: fractional {column} values")
        df[column] = df[column].astype("Int64")
    original_time = df["Date_Time"].copy()
    df["Date_Time"] = pd.to_datetime(original_time, errors="coerce")
    if (original_time.notna() & df["Date_Time"].isna()).any():
        raise RuntimeError(f"{source.name}: unparseable Date_Time values")
    if df["Cycle_Index"].isna().any():
        raise RuntimeError(f"{source.name}: Cycle_Index contains missing values")
    return df


def weighted_quantile(values: pd.Series, weights: pd.Series, quantile: float) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    value_array = values.loc[valid].to_numpy(dtype=float)
    weight_array = weights.loc[valid].to_numpy(dtype=float)
    order = np.argsort(value_array)
    value_array, weight_array = value_array[order], weight_array[order]
    cumulative = np.cumsum(weight_array)
    cutoff = quantile * cumulative[-1]
    return float(value_array[min(np.searchsorted(cumulative, cutoff, side="left"), len(value_array) - 1)])


def add_row_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    current = df["Current(A)"]
    df["_raw_phase"] = np.select(
        [current.isna(), current > REST_THRESHOLD_A, current < -REST_THRESHOLD_A],
        ["UNKNOWN", "CHARGE", "DISCHARGE"], default="REST",
    )
    df["_dt_test"] = df["Test_Time(s)"].diff()
    df["_dt_timestamp"] = df["Date_Time"].diff().dt.total_seconds()
    df["_same_cycle"] = df["Cycle_Index"].eq(df["Cycle_Index"].shift())
    raw_transition = ~df["_same_cycle"] | df["_raw_phase"].ne(df["_raw_phase"].shift())
    df["_raw_phase_id"] = raw_transition.cumsum().astype("int64")

    cycle_groups = df.groupby("Cycle_Index", sort=False, dropna=False)
    previous_current = cycle_groups["Current(A)"].shift()
    next_current = cycle_groups["Current(A)"].shift(-1)
    previous_voltage = cycle_groups["Voltage(V)"].shift()
    next_voltage = cycle_groups["Voltage(V)"].shift(-1)
    previous_phase = cycle_groups["_raw_phase"].shift()
    next_phase = cycle_groups["_raw_phase"].shift(-1)

    rolling_median = df.groupby("_raw_phase_id", sort=False)["Current(A)"].transform(
        lambda series: series.rolling(HAMPEL_WINDOW, center=True, min_periods=3).median()
    )
    residual = (df["Current(A)"] - rolling_median).abs()
    rolling_mad = residual.groupby(df["_raw_phase_id"], sort=False).transform(
        lambda series: series.rolling(HAMPEL_WINDOW, center=True, min_periods=3).median()
    )
    hampel_threshold = (HAMPEL_SIGMA * 1.4826 * rolling_mad).clip(lower=CURRENT_JUMP_THRESHOLD_A)
    df["_hampel_candidate"] = residual.gt(hampel_threshold) & rolling_median.notna()
    df["_isolated_candidate"] = (
        previous_current.notna() & next_current.notna()
        & previous_current.sub(next_current).abs().le(NEIGHBOUR_TOLERANCE_A)
        & df["Current(A)"].sub(previous_current).abs().gt(CURRENT_JUMP_THRESHOLD_A)
        & df["Current(A)"].sub(next_current).abs().gt(CURRENT_JUMP_THRESHOLD_A)
    )
    df["_artifact_candidate"] = df["_hampel_candidate"] | df["_isolated_candidate"]

    delta_i = df["Current(A)"].diff()
    delta_v = df["Voltage(V)"].diff()
    transition_mask = (
        df["_same_cycle"] & df["_dt_test"].gt(0) & df["_dt_test"].le(MAX_INTERVAL_SECONDS)
        & delta_i.abs().gt(CURRENT_JUMP_THRESHOLD_A) & delta_v.notna()
    )
    resistance = (delta_v.loc[transition_mask] / delta_i.loc[transition_mask]).abs()
    resistance = resistance.replace([np.inf, -np.inf], np.nan).dropna()
    if resistance.empty:
        resistance_median = resistance_lower = resistance_upper = np.nan
    else:
        resistance_median = float(resistance.median())
        resistance_mad = float((resistance - resistance_median).abs().median())
        resistance_scale = 1.4826 * resistance_mad
        if resistance_scale == 0:
            resistance_lower = 0.0
            resistance_upper = max(resistance_median * 3.0, resistance_median + 1e-6)
        else:
            resistance_lower = max(0.0, resistance_median - HAMPEL_SIGMA * resistance_scale)
            resistance_upper = resistance_median + HAMPEL_SIGMA * resistance_scale

    neighbour_current = (previous_current + next_current) / 2.0
    neighbour_voltage = (previous_voltage + next_voltage) / 2.0
    candidate_di = (df["Current(A)"] - neighbour_current).abs()
    candidate_dv = (df["Voltage(V)"] - neighbour_voltage).abs()
    candidate_resistance = candidate_dv / candidate_di.replace(0, np.nan)
    has_voltage_evidence = df["Voltage(V)"].notna() & previous_voltage.notna() & next_voltage.notna()
    voltage_coherent = (
        candidate_resistance.ge(resistance_lower) & candidate_resistance.le(resistance_upper)
        if np.isfinite(resistance_lower) and np.isfinite(resistance_upper)
        else pd.Series(False, index=df.index)
    )

    session_sizes = df.groupby("_raw_phase_id", sort=False)["Current(A)"].transform("size")
    protocol_current = (
        df["_raw_phase"].eq("CHARGE")
        & df["Current(A)"].gt(RATED_CHARGE_CURRENT_A)
        & session_sizes.ge(MINIMUM_PERSISTENT_SAMPLES)
    )
    df["_artifact_class"] = "NONE"
    df.loc[protocol_current, "_artifact_class"] = "REAL_PROTOCOL_CURRENT"
    confirmed_artifact = (
        df["_isolated_candidate"] & df["_hampel_candidate"]
        & has_voltage_evidence & ~voltage_coherent
    )
    real_transient = df["_artifact_candidate"] & voltage_coherent & ~confirmed_artifact
    review_required = df["_artifact_candidate"] & ~confirmed_artifact & ~real_transient
    df.loc[real_transient, "_artifact_class"] = "REAL_TRANSIENT"
    df.loc[review_required, "_artifact_class"] = "REVIEW_REQUIRED"
    df.loc[confirmed_artifact, "_artifact_class"] = "LIKELY_SENSOR_ARTIFACT"
    df["_current_artifact"] = confirmed_artifact

    # Bridge only the temporary phase label across a confirmed isolated artifact.
    bridge = confirmed_artifact & previous_phase.eq(next_phase) & previous_phase.notna()
    df["_phase"] = df["_raw_phase"]
    df.loc[bridge, "_phase"] = previous_phase.loc[bridge]
    transition = ~df["_same_cycle"] | df["_phase"].ne(df["_phase"].shift())
    df["_phase_id"] = transition.cumsum().astype("int64")
    df["_same_phase"] = df["_phase"].eq(df["_phase"].shift())
    df["_timestamp_reversal"] = df["_same_cycle"] & df["_dt_timestamp"].lt(0)
    df["_current_previous"] = df["Current(A)"].shift()
    df["_voltage_previous"] = df["Voltage(V)"].shift()
    df["_capacity_previous"] = df["Capacity(Ah)"].shift()
    df["_energy_previous"] = df["Energy(Wh)"].shift()
    df["_artifact_previous"] = df["_current_artifact"].shift(fill_value=False)
    candidate = df["_same_cycle"] & df["_same_phase"]
    valid_time = df["_dt_test"].gt(0) & df["_dt_test"].le(MAX_INTERVAL_SECONDS)
    current_present = df[["Current(A)", "_current_previous"]].notna().all(axis=1)
    voltage_present = df[["Voltage(V)", "_voltage_previous"]].notna().all(axis=1)
    artifact_free = ~df["_current_artifact"] & ~df["_artifact_previous"]
    df["_valid_capacity_interval"] = (
        candidate & valid_time & ~df["_timestamp_reversal"] & current_present & artifact_free
    )
    df["_valid_energy_interval"] = df["_valid_capacity_interval"] & voltage_present
    invalid_data_or_time = (
        df["_dt_test"].le(0) | df["_dt_test"].gt(MAX_INTERVAL_SECONDS)
        | df["_dt_test"].isna() | df["_timestamp_reversal"]
        | ~current_present | ~voltage_present | ~artifact_free
    )
    df["_invalid_interval"] = candidate & invalid_data_or_time
    df.attrs["resistance_evidence"] = {
        "transition_samples": int(len(resistance)),
        "median_ohm": resistance_median,
        "lower_ohm": resistance_lower,
        "upper_ohm": resistance_upper,
    }
    return df


def session_counter_total(group: pd.DataFrame, phase: str, column: str) -> float:
    total = 0.0
    found = False
    phase_rows = group[group["_phase"].eq(phase)]
    for _, session in phase_rows.groupby("_phase_id", sort=False):
        values = session[column].dropna()
        if len(values) >= 2:
            total += float(values.max() - values.min())
            found = True
    return total if found else np.nan


def relative_mismatch(integrated: float, counter: float) -> bool:
    if not np.isfinite(integrated) or not np.isfinite(counter) or counter <= 0:
        return False
    return abs(integrated - counter) / counter > COUNTER_RELATIVE_TOLERANCE


def integrate_phase(group: pd.DataFrame, phase: str):
    capacity_rows = group[group["_valid_capacity_interval"] & group["_phase"].eq(phase)]
    energy_rows = group[group["_valid_energy_interval"] & group["_phase"].eq(phase)]
    if capacity_rows.empty:
        capacity_ah = np.nan
        duration_s = 0.0
    else:
        interval_current = (
            capacity_rows["Current(A)"].abs() + capacity_rows["_current_previous"].abs()
        ) / 2.0
        capacity_ah = float((interval_current * capacity_rows["_dt_test"]).sum() / 3600.0)
        duration_s = float(capacity_rows["_dt_test"].sum())
    if energy_rows.empty:
        energy_wh = np.nan
    else:
        power_now = energy_rows["Voltage(V)"] * energy_rows["Current(A)"].abs()
        power_previous = energy_rows["_voltage_previous"] * energy_rows["_current_previous"].abs()
        energy_wh = float((((power_now + power_previous) / 2.0) * energy_rows["_dt_test"]).sum() / 3600.0)
    return capacity_rows, capacity_ah, duration_s, energy_wh


def cycle_features(dataset_id: str, cycle_index: int, group: pd.DataFrame):
    charge_rows, charge_capacity, charge_duration, charge_energy = integrate_phase(group, "CHARGE")
    discharge_rows, discharge_capacity, discharge_duration, discharge_energy = integrate_phase(group, "DISCHARGE")
    _, _, rest_duration, _ = integrate_phase(group, "REST")

    phase_counts = group.groupby("_phase")["_phase_id"].nunique().to_dict()
    charge_count = int(phase_counts.get("CHARGE", 0))
    discharge_count = int(phase_counts.get("DISCHARGE", 0))
    rest_count = int(phase_counts.get("REST", 0))
    charge_present = charge_count > 0
    discharge_present = discharge_count > 0

    charge_invalid_major = bool(
        group.loc[group["_phase"].eq("CHARGE"), "Current(A)"].isna().any()
        or group.loc[group["_phase"].eq("CHARGE"), "Voltage(V)"].isna().any()
        or (group["_phase"].eq("CHARGE") & group["_dt_test"].gt(MAX_INTERVAL_SECONDS)).any()
        or (group["_phase"].eq("CHARGE") & group["_dt_test"].lt(0)).any()
    )
    discharge_invalid_major = bool(
        group.loc[group["_phase"].eq("DISCHARGE"), "Current(A)"].isna().any()
        or group.loc[group["_phase"].eq("DISCHARGE"), "Voltage(V)"].isna().any()
        or (group["_phase"].eq("DISCHARGE") & group["_dt_test"].gt(MAX_INTERVAL_SECONDS)).any()
        or (group["_phase"].eq("DISCHARGE") & group["_dt_test"].lt(0)).any()
    )
    charge_valid = bool(
        charge_present and not charge_rows.empty and np.isfinite(charge_capacity)
        and charge_capacity > 0 and np.isfinite(charge_energy) and not charge_invalid_major
    )
    discharge_valid = bool(
        discharge_present and not discharge_rows.empty and np.isfinite(discharge_capacity)
        and discharge_capacity > 0 and np.isfinite(discharge_energy) and not discharge_invalid_major
    )
    complete = charge_valid and discharge_valid

    if complete:
        cycle_status = "COMPLETE"
    elif charge_present and not discharge_present:
        cycle_status = "PARTIAL_CHARGE"
    elif discharge_present and not charge_present:
        cycle_status = "PARTIAL_DISCHARGE"
    else:
        cycle_status = "INVALID"

    if charge_rows.empty:
        mean_charge_c = p99_charge_c = time_above_1c = np.nan
    else:
        interval_c = (
            charge_rows["Current(A)"].abs() + charge_rows["_current_previous"].abs()
        ) / (2.0 * NOMINAL_CAPACITY_AH)
        interval_time = charge_rows["_dt_test"]
        mean_charge_c = float((interval_c * interval_time).sum() / interval_time.sum())
        p99_charge_c = weighted_quantile(interval_c, interval_time, 0.99)
        time_above_1c = float(interval_time.loc[interval_c.gt(1.0)].sum())

    ce = float(discharge_capacity / charge_capacity * 100.0) if complete else np.nan
    throughput = (
        float(charge_energy + discharge_energy)
        if np.isfinite(charge_energy) and np.isfinite(discharge_energy) else np.nan
    )
    valid_elapsed = group.loc[
        group["_same_cycle"]
        & group["_dt_test"].gt(0)
        & group["_dt_test"].le(MAX_INTERVAL_SECONDS)
        & ~group["_timestamp_reversal"],
        "_dt_test",
    ]
    cycle_duration = float(valid_elapsed.sum())

    duplicated_times = group.loc[
        group["Date_Time"].notna() & group["Date_Time"].duplicated(keep=False), "Date_Time"
    ].nunique()
    reversal_count = int(group["_timestamp_reversal"].sum())
    timestamp_issues = int(duplicated_times + reversal_count)
    invalid_intervals = int(group["_invalid_interval"].sum())

    charge_counter = session_counter_total(group, "CHARGE", "Capacity(Ah)")
    discharge_counter = session_counter_total(group, "DISCHARGE", "Capacity(Ah)")
    charge_energy_counter = session_counter_total(group, "CHARGE", "Energy(Wh)")
    discharge_energy_counter = session_counter_total(group, "DISCHARGE", "Energy(Wh)")

    flags = []
    if not charge_present:
        flags.append("NO_CHARGE_PHASE")
    if not discharge_present:
        flags.append("NO_DISCHARGE_PHASE")
    if reversal_count:
        flags.append("TIMESTAMP_REVERSAL_OMITTED")
    if (group["_dt_test"].eq(0) & group["_same_cycle"] & group["_same_phase"]).any():
        flags.append("ZERO_DURATION_INTERVAL_OMITTED")
    if (group["_dt_test"].gt(MAX_INTERVAL_SECONDS) & group["_same_cycle"]).any():
        flags.append("SAMPLING_GAP")
    if group["Current(A)"].isna().any():
        flags.append("MISSING_CURRENT")
    if group["Voltage(V)"].isna().any():
        flags.append("MISSING_VOLTAGE")
    if charge_count > 1:
        flags.append("MULTIPLE_CHARGE_PHASES")
    if discharge_count > 1:
        flags.append("MULTIPLE_DISCHARGE_PHASES")
    artifact_count = int(group["_current_artifact"].sum())
    artifact_interval_count = int(
        (group["_current_artifact"] | group["_artifact_previous"]).mul(group["_same_cycle"] & group["_same_phase"]).sum()
    )
    if artifact_count:
        flags.append("CURRENT_SENSOR_ARTIFACT")
    if relative_mismatch(charge_capacity, charge_counter) or relative_mismatch(discharge_capacity, discharge_counter):
        flags.append("CAPACITY_COUNTER_MISMATCH")
    if relative_mismatch(charge_energy, charge_energy_counter) or relative_mismatch(discharge_energy, discharge_energy_counter):
        flags.append("ENERGY_COUNTER_MISMATCH")
    flags = list(dict.fromkeys(flags))

    if not complete:
        quality_status = "INVALID"
    elif any(flag not in {"ZERO_DURATION_INTERVAL_OMITTED"} for flag in flags):
        quality_status = "WARNING"
    else:
        quality_status = "PASS"

    record = {
        "Dataset_ID": dataset_id,
        "Cycle_Index": int(cycle_index),
        "Cycle_Status": cycle_status,
        "Cycle_Complete": complete,
        "Charge_Phase_Count": charge_count,
        "Discharge_Phase_Count": discharge_count,
        "Rest_Phase_Count": rest_count,
        "Mean_C_Rate_Charge": mean_charge_c,
        "P99_C_Rate_Charge": p99_charge_c,
        "Time_Above_1C_Charge_s": time_above_1c,
        "Charge_Capacity_Ah": charge_capacity,
        "Discharge_Capacity_Ah": discharge_capacity,
        "Coulombic_Efficiency_pct": ce,
        "Charge_Duration_s": charge_duration,
        "Discharge_Duration_s": discharge_duration,
        "Rest_Duration_s": rest_duration,
        "Cycle_Duration_s": cycle_duration,
        "Charge_Energy_Wh": charge_energy,
        "Discharge_Energy_Wh": discharge_energy,
        "Energy_Throughput_Wh": throughput,
        "Charge_Integration_Valid": charge_valid,
        "Discharge_Integration_Valid": discharge_valid,
        "Timestamp_Issue_Count": timestamp_issues,
        "Invalid_Interval_Count": invalid_intervals,
        "Quality_Flag_Count": len(flags),
        "Quality_Status": quality_status,
    }
    diagnostics = {
        "Dataset_ID": dataset_id,
        "Cycle_Index": int(cycle_index),
        "Quality_Flags": flags,
        "Charge_Counter_Ah": charge_counter,
        "Discharge_Counter_Ah": discharge_counter,
        "Charge_Energy_Counter_Wh": charge_energy_counter,
        "Discharge_Energy_Counter_Wh": discharge_energy_counter,
        "Candidate_Current_Spike_Count": int(group["_artifact_candidate"].sum()),
        "Likely_Artifact_Count": artifact_count,
        "Real_Transient_Count": int(group["_artifact_class"].eq("REAL_TRANSIENT").sum()),
        "Review_Required_Count": int(group["_artifact_class"].eq("REVIEW_REQUIRED").sum()),
        "Protocol_Current_Row_Count": int(group["_artifact_class"].eq("REAL_PROTOCOL_CURRENT").sum()),
        "Excluded_Artifact_Interval_Count": artifact_interval_count,
    }
    return record, diagnostics


def validate_gold(gold: pd.DataFrame, source_cycle_counts: dict[str, int]):
    failures = []
    if list(gold.columns) != GOLD_COLUMNS:
        failures.append("output column order/schema mismatch")
    if gold.duplicated(["Dataset_ID", "Cycle_Index"]).any():
        failures.append("duplicate Dataset_ID + Cycle_Index keys")
    for dataset_id, expected in source_cycle_counts.items():
        actual = gold.loc[gold["Dataset_ID"].eq(dataset_id), "Cycle_Index"].nunique()
        if actual != expected:
            failures.append(f"{dataset_id}: expected {expected} cycles, found {actual}")
    nonnegative = [
        "Mean_C_Rate_Charge", "P99_C_Rate_Charge", "Time_Above_1C_Charge_s",
        "Charge_Capacity_Ah", "Discharge_Capacity_Ah", "Coulombic_Efficiency_pct",
        "Charge_Duration_s", "Discharge_Duration_s", "Rest_Duration_s", "Cycle_Duration_s",
        "Charge_Energy_Wh", "Discharge_Energy_Wh", "Energy_Throughput_Wh",
        "Timestamp_Issue_Count", "Invalid_Interval_Count", "Quality_Flag_Count",
    ]
    for column in nonnegative:
        if gold[column].dropna().lt(0).any():
            failures.append(f"negative values in {column}")
    ce_allowed = gold["Cycle_Complete"] & gold["Charge_Integration_Valid"] & gold["Discharge_Integration_Valid"]
    if gold.loc[~ce_allowed, "Coulombic_Efficiency_pct"].notna().any():
        failures.append("CE present for incomplete or invalid cycle")
    if gold.loc[ce_allowed, "Coulombic_Efficiency_pct"].isna().any():
        failures.append("CE missing for complete valid cycle")
    if np.isinf(gold.select_dtypes(include=[np.number]).to_numpy()).any():
        failures.append("infinite numeric value")
    if failures:
        raise RuntimeError("Gold validation failed: " + "; ".join(failures))


def report_markdown(gold: pd.DataFrame, diagnostics: list[dict], output_path: Path) -> str:
    lines = [
        "# Gold Cycle Feature Report", "",
        f"Output: `{output_path.name}`", "",
        "## Method", "",
        "- One row per Dataset_ID and Cycle_Index.",
        "- Nominal capacity: 2.5 Ah.",
        "- Charge threshold: current > 0.05 A; discharge threshold: current < -0.05 A.",
        "- Test_Time(s) is the integration axis.",
        "- Integration is trapezoidal and remains within one cycle and one phase session.",
        "- Zero/negative intervals, intervals above 5 s, timestamp reversals, and missing endpoint data are omitted.",
        "- Confirmed current-sensor artifacts are flagged and their touching intervals are omitted; persistent protocol currents are retained.",
        "- Charge-current extremity is summarized with a duration-weighted P99 rather than a bare maximum.",
        "- Cycles 1 and 101 are retained but do not receive coulombic efficiency.", "",
    ]
    for dataset_id, group in gold.groupby("Dataset_ID", sort=False):
        lines += [
            f"## Dataset {dataset_id}", "",
            f"- Cycles: **{len(group)}**",
            f"- Complete cycles: **{int(group['Cycle_Complete'].sum())}**",
            f"- Partial/invalid cycles: **{int((~group['Cycle_Complete']).sum())}**",
            f"- PASS / WARNING / INVALID: **{int(group['Quality_Status'].eq('PASS').sum())} / {int(group['Quality_Status'].eq('WARNING').sum())} / {int(group['Quality_Status'].eq('INVALID').sum())}**",
            f"- Median discharge capacity: **{group['Discharge_Capacity_Ah'].median():.6f} Ah**",
            f"- Median coulombic efficiency among valid cycles: **{group['Coulombic_Efficiency_pct'].median():.4f}%**", "",
        ]
    flag_counts = {}
    for item in diagnostics:
        for flag in item["Quality_Flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    lines += ["## Cycle-level quality reasons", "", "| Flag | Cycles |", "|---|---:|"]
    for flag, count in sorted(flag_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {flag} | {count} |")
    lines += ["", "## Validation", "",
              "- Source cycle counts reconcile to the Gold table.",
              "- Dataset_ID + Cycle_Index is unique.",
              "- All calculated physical metrics are nonnegative.",
              "- Coulombic efficiency exists only for complete valid cycles.",
              "- The Gold table contains exactly 26 columns.", ""]
    return "\n".join(lines)


def artifact_report_markdown(diagnostics: list[dict], resistance_evidence: dict) -> str:
    lines = [
        "# Current Artifact Diagnostics", "",
        "Current candidates were assessed with persistence, a session-local Hampel identifier, and voltage/current coherence. "
        "Values were never overwritten. Only intervals touching a confirmed likely sensor artifact are excluded from Gold calculations.", "",
    ]
    for dataset_id in ["001_2", "002_4"]:
        items = [item for item in diagnostics if item["Dataset_ID"] == dataset_id]
        totals = {
            key: sum(int(item[key]) for item in items)
            for key in [
                "Candidate_Current_Spike_Count", "Likely_Artifact_Count", "Real_Transient_Count",
                "Review_Required_Count", "Protocol_Current_Row_Count", "Excluded_Artifact_Interval_Count",
            ]
        }
        evidence = resistance_evidence[dataset_id]
        resistance_text = (
            f"{evidence['median_ohm']} Ω"
            if evidence["transition_samples"] and np.isfinite(evidence["median_ohm"])
            else "Unavailable (no qualifying transition samples)"
        )
        lines += [
            f"## Dataset {dataset_id}", "",
            f"- Candidate current rows: **{totals['Candidate_Current_Spike_Count']:,}**",
            f"- Likely sensor artifacts: **{totals['Likely_Artifact_Count']:,}**",
            f"- Real transients: **{totals['Real_Transient_Count']:,}**",
            f"- Review required: **{totals['Review_Required_Count']:,}**",
            f"- Persistent protocol-current rows: **{totals['Protocol_Current_Row_Count']:,}**",
            f"- Excluded artifact-touching intervals: **{totals['Excluded_Artifact_Interval_Count']:,}**",
            f"- Transition resistance samples: **{evidence['transition_samples']:,}**",
            f"- Robust median apparent transition resistance: **{resistance_text}**", "",
        ]
    lines += [
        "## Interpretation", "",
        "- Persistent charge-current plateaus above the provisional rating are classified as experimental protocol currents, not sensor spikes.",
        "- A candidate is confirmed as a likely artifact only when it is isolated, Hampel-flagged, and voltage-incoherent.",
        "- Voltage-coherence classification cannot be calibrated when a dataset has no qualifying nonzero-time current-step samples; ambiguous candidates would remain `REVIEW_REQUIRED`.",
        "- Ambiguous candidates remain in the data as `REVIEW_REQUIRED` and are not automatically excluded.", "",
    ]
    return "\n".join(lines)


def legacy_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-001-2", type=Path, default=DEFAULT_INPUTS["001_2"])
    parser.add_argument("--input-002-4", type=Path, default=DEFAULT_INPUTS["002_4"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    inputs = {"001_2": args.input_001_2, "002_4": args.input_002_4}

    records, diagnostics, source_cycle_counts, resistance_evidence = [], [], {}, {}
    for dataset_id, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(path)
        raw = pd.read_parquet(path)
        source_rows = len(raw)
        df = add_row_state(validate_and_standardize(raw, path))
        resistance_evidence[dataset_id] = df.attrs["resistance_evidence"]
        if len(df) != source_rows:
            raise RuntimeError(f"{path.name}: source row count changed")
        source_cycle_counts[dataset_id] = int(df["Cycle_Index"].nunique())
        for cycle_index, group in df.groupby("Cycle_Index", sort=True, dropna=False):
            record, diagnostic = cycle_features(dataset_id, int(cycle_index), group.copy())
            records.append(record)
            diagnostics.append(diagnostic)

    gold = pd.DataFrame(records, columns=GOLD_COLUMNS).sort_values(
        ["Dataset_ID", "Cycle_Index"], kind="stable"
    ).reset_index(drop=True)
    validate_gold(gold, source_cycle_counts)

    args.out.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out / "battery_cycle_features_gold.parquet"
    report_path = args.out / "cycle_feature_report.md"
    summary_path = args.out / "cycle_feature_summary.json"
    artifact_report_path = args.out / "current_artifact_report.md"
    artifact_json_path = args.out / "current_artifact_diagnostics.json"
    gold.to_parquet(parquet_path, index=False, compression="snappy")
    report_path.write_text(report_markdown(gold, diagnostics, parquet_path), encoding="utf-8")
    artifact_report_path.write_text(
        artifact_report_markdown(diagnostics, resistance_evidence), encoding="utf-8"
    )
    artifact_json_path.write_text(json.dumps({
        "configuration": CONFIG,
        "resistance_evidence": resistance_evidence,
        "cycle_artifact_diagnostics": diagnostics,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    summary_path.write_text(json.dumps({
        "configuration": {
            "nominal_capacity_ah": NOMINAL_CAPACITY_AH,
            "rest_threshold_a": REST_THRESHOLD_A,
            "maximum_interval_seconds": MAX_INTERVAL_SECONDS,
            "counter_relative_tolerance": COUNTER_RELATIVE_TOLERANCE,
            "hampel_window_samples": HAMPEL_WINDOW,
            "hampel_sigma": HAMPEL_SIGMA,
        },
        "source_cycle_counts": source_cycle_counts,
        "gold_rows": len(gold),
        "gold_columns": GOLD_COLUMNS,
        "cycle_diagnostics": diagnostics,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Created {parquet_path} ({len(gold)} rows, {len(gold.columns)} columns)")
    print(f"Created {report_path}")
    print(f"Created {summary_path}")
    print(f"Created {artifact_report_path}")
    print(f"Created {artifact_json_path}")


if __name__ == "__main__":
    from battery_validated_pipeline import main
    main()
