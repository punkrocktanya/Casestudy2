#!/usr/bin/env python3
"""Create validated Silver battery datasets from cleaned Parquet inputs.

The transformation preserves source row order and measurements. It does not
delete, sort, interpolate, clip, round, or repair source observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from battery_config import CONFIG, CONFIG_SHA256


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUTS = [
    ROOT / "outputs" / "battery_cleaning" / "LR1865SZ_cycles201217_001_2_clean.parquet",
    ROOT / "outputs" / "battery_cleaning" / "LR1865SZ_cycles201214_002_4_clean.parquet",
]
DEFAULT_OUTPUT = ROOT / "outputs" / "silver"
REQUIRED_COLUMNS = [
    "Data_Point", "Test_Time(s)", "Current(A)", "Capacity(Ah)",
    "Voltage(V)", "Energy(Wh)", "Temperature(℃)", "Date_Time", "Cycle_Index",
]
NUMERIC_COLUMNS = [
    "Data_Point", "Test_Time(s)", "Current(A)", "Capacity(Ah)",
    "Voltage(V)", "Energy(Wh)", "Temperature(℃)", "Cycle_Index",
]
REST_THRESHOLD_A = CONFIG['rest_current_threshold_a']
SAMPLING_GAP_SECONDS = CONFIG['maximum_integration_interval_s']
MAX_CHARGE_CURRENT_A = CONFIG['rated_charge_current_a']
MAX_DISCHARGE_CURRENT_A = CONFIG['rated_discharge_current_a']
VOLTAGE_MIN_V = CONFIG['voltage_min_v']
VOLTAGE_MAX_V = CONFIG['voltage_max_v']
CAPACITY_MAX_AH = CONFIG['capacity_warning_max_ah']
ENERGY_MAX_WH = CONFIG['energy_warning_max_wh']


def validate_schema(raw: pd.DataFrame, source: Path) -> None:
    duplicate_columns = raw.columns[raw.columns.duplicated()].tolist()
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if duplicate_columns or missing:
        raise RuntimeError(
            f"{source.name}: schema failure; missing={missing}, duplicate_columns={duplicate_columns}"
        )
    if raw.empty:
        raise RuntimeError(f"{source.name}: input contains zero rows")


def numeric_conversion(raw: pd.DataFrame, source: Path):
    converted = raw.loc[:, REQUIRED_COLUMNS].copy()
    report = {}
    for column in NUMERIC_COLUMNS:
        original = converted[column]
        sentinels = ["", "-", "--", "NA", "N/A", "null", "None"]
        if column == "Temperature(℃)":
            sentinels += ["-999", "65535", "32767"]
        cleaned = original.astype("string").str.strip().replace(sentinels, pd.NA)
        numeric = pd.to_numeric(cleaned, errors="coerce")
        failures = original.notna() & numeric.isna()
        report[column] = {
            "conversion_failures": int(failures.sum()),
            "missing_after_conversion": int(numeric.isna().sum()),
        }
        if failures.any() and column != "Temperature(℃)":
            examples = original.loc[failures].astype(str).head(5).tolist()
            raise RuntimeError(f"{source.name}: {column} conversion failures; examples={examples}")
        converted[column] = numeric

    for column in ["Data_Point", "Cycle_Index"]:
        fractional = converted[column].notna() & converted[column].mod(1).ne(0)
        if fractional.any():
            raise RuntimeError(f"{source.name}: {column} contains fractional values")
        converted[column] = converted[column].astype("Int64")

    original_time = converted["Date_Time"].copy()
    parsed_time = pd.to_datetime(original_time, errors="coerce")
    parse_failures = original_time.notna() & parsed_time.isna()
    report["Date_Time"] = {
        "conversion_failures": int(parse_failures.sum()),
        "missing_after_conversion": int(parsed_time.isna().sum()),
    }
    if parse_failures.any():
        raise RuntimeError(f"{source.name}: Date_Time contains unparseable values")
    converted["Date_Time"] = parsed_time
    return converted, report


def classify_and_segment(df: pd.DataFrame) -> pd.DataFrame:
    current = df["Current(A)"]
    df["Current_Abs(A)"] = current.abs()
    df["Phase"] = np.select(
        [current.isna(), current > REST_THRESHOLD_A, current < -REST_THRESHOLD_A],
        ["UNKNOWN", "CHARGE", "DISCHARGE"],
        default="REST",
    )
    previous_cycle = df["Cycle_Index"].shift()
    cycle_changed = df["Cycle_Index"].ne(previous_cycle) | (df["Cycle_Index"].isna() != previous_cycle.isna())
    df["Phase_Transition"] = (
        df["Phase"].ne(df["Phase"].shift()) | cycle_changed | df.index.to_series().eq(df.index[0])
    )
    df["Phase_ID"] = df["Phase_Transition"].cumsum().astype("Int64")
    df["Phase_Sequence"] = (
        df.groupby("Cycle_Index", dropna=False, sort=False)["Phase_Transition"].cumsum().astype("Int64")
    )
    return df


def sign_convention_evidence(df: pd.DataFrame):
    segment_rows = []
    for phase_id, group in df.groupby("Phase_ID", sort=False):
        phase = group["Phase"].iloc[0]
        voltage = group["Voltage(V)"].dropna()
        if phase not in {"CHARGE", "DISCHARGE"} or len(voltage) < 2:
            continue
        segment_rows.append({
            "phase_id": int(phase_id),
            "phase": phase,
            "voltage_change_v": float(voltage.iloc[-1] - voltage.iloc[0]),
        })
    segments = pd.DataFrame(segment_rows)
    result = {}
    for phase, expected_positive in [("CHARGE", True), ("DISCHARGE", False)]:
        values = segments.loc[segments["phase"].eq(phase), "voltage_change_v"] if not segments.empty else pd.Series(dtype=float)
        if values.empty:
            result[phase.lower()] = {
                "segments_checked": 0, "median_segment_voltage_change_v": None,
                "expected_direction_percentage": None,
            }
            continue
        agrees = values.gt(0) if expected_positive else values.lt(0)
        result[phase.lower()] = {
            "segments_checked": int(len(values)),
            "median_segment_voltage_change_v": float(values.median()),
            "expected_direction_percentage": float(100 * agrees.mean()),
        }
    charge = result["charge"]
    discharge = result["discharge"]
    result["positive_current_is_charge_supported"] = bool(
        charge["segments_checked"] > 0
        and discharge["segments_checked"] > 0
        and charge["median_segment_voltage_change_v"] > 0
        and discharge["median_segment_voltage_change_v"] < 0
    )
    return result


def transform(raw: pd.DataFrame, source: Path):
    validate_schema(raw, source)
    original_rows = len(raw)
    original_order = raw["Data_Point"].copy()
    df, conversions = numeric_conversion(raw, source)
    df.insert(0, "Row_ID", np.arange(1, len(df) + 1, dtype=np.int64))

    df["Test_Time_Delta_s"] = df["Test_Time(s)"].diff()
    df["Date_Time_Delta_s"] = df["Date_Time"].diff().dt.total_seconds()
    df = classify_and_segment(df)

    df["Critical_Missing_Flag"] = df[
        ["Test_Time(s)", "Current(A)", "Voltage(V)", "Date_Time", "Cycle_Index"]
    ].isna().any(axis=1)
    df["Capacity_Missing_Flag"] = df["Capacity(Ah)"].isna()
    df["Energy_Missing_Flag"] = df["Energy(Wh)"].isna()
    df["Temperature_Missing_Flag"] = df["Temperature(℃)"].isna()
    df["Missing_Flag"] = df[
        ["Critical_Missing_Flag", "Capacity_Missing_Flag", "Energy_Missing_Flag", "Temperature_Missing_Flag"]
    ].any(axis=1)
    df["Temperature_Status"] = np.where(df["Temperature_Missing_Flag"], "MISSING", "OBSERVED")
    df["Timestamp_Duplicate_Flag"] = df["Date_Time"].notna() & df["Date_Time"].duplicated(keep=False)
    df["Timestamp_Reversal_Flag"] = df["Date_Time_Delta_s"].lt(0)
    df["Test_Time_Reversal_Flag"] = df["Test_Time_Delta_s"].lt(0)
    df["Sampling_Gap_Flag"] = df["Test_Time_Delta_s"].gt(SAMPLING_GAP_SECONDS)
    df["Current_Plausibility_Flag"] = (
        (df["Phase"].eq("CHARGE") & df["Current(A)"].gt(MAX_CHARGE_CURRENT_A))
        | (df["Phase"].eq("DISCHARGE") & df["Current(A)"].abs().gt(MAX_DISCHARGE_CURRENT_A))
    )
    df["Voltage_Plausibility_Flag"] = df["Voltage(V)"].notna() & ~df["Voltage(V)"].between(
        VOLTAGE_MIN_V, VOLTAGE_MAX_V
    )
    df["Capacity_Plausibility_Flag"] = df["Capacity(Ah)"].notna() & ~df["Capacity(Ah)"].between(
        0, CAPACITY_MAX_AH
    )
    df["Energy_Plausibility_Flag"] = df["Energy(Wh)"].notna() & ~df["Energy(Wh)"].between(
        0, ENERGY_MAX_WH
    )

    output_columns = [
        "Row_ID", "Data_Point", "Test_Time(s)", "Current(A)", "Current_Abs(A)",
        "Capacity(Ah)", "Voltage(V)", "Energy(Wh)", "Temperature(℃)", "Temperature_Status",
        "Date_Time", "Cycle_Index", "Test_Time_Delta_s", "Date_Time_Delta_s", "Phase",
        "Phase_ID", "Phase_Sequence", "Phase_Transition", "Missing_Flag",
        "Critical_Missing_Flag", "Capacity_Missing_Flag", "Energy_Missing_Flag",
        "Temperature_Missing_Flag", "Timestamp_Duplicate_Flag", "Timestamp_Reversal_Flag",
        "Test_Time_Reversal_Flag", "Sampling_Gap_Flag", "Current_Plausibility_Flag",
        "Voltage_Plausibility_Flag", "Capacity_Plausibility_Flag", "Energy_Plausibility_Flag",
    ]
    df = df.loc[:, output_columns]

    allowed_phases = {"CHARGE", "DISCHARGE", "REST", "UNKNOWN"}
    expected_phase_ids = np.arange(1, int(df["Phase_ID"].max()) + 1)
    phase_ids = df["Phase_ID"].drop_duplicates().astype(int).to_numpy()
    measurement_equivalence = {}
    for column in REQUIRED_COLUMNS:
        if column == "Date_Time":
            expected = pd.to_datetime(raw[column], errors="coerce")
            measurement_equivalence[column] = bool(expected.equals(df[column]))
        elif column in ["Data_Point", "Cycle_Index"]:
            expected = pd.to_numeric(raw[column], errors="coerce").astype("Int64")
            measurement_equivalence[column] = bool(expected.equals(df[column]))
        else:
            sentinels = ["", "-", "--", "NA", "N/A", "null", "None"]
            if column == "Temperature(℃)":
                sentinels += ["-999", "65535", "32767"]
            expected = pd.to_numeric(
                raw[column].astype("string").str.strip().replace(sentinels, pd.NA), errors="coerce"
            )
            measurement_equivalence[column] = bool(np.allclose(expected, df[column], equal_nan=True))

    validations = {
        "row_count_preserved": len(df) == original_rows,
        "row_order_preserved": bool(np.allclose(
            pd.to_numeric(original_order, errors="coerce"),
            df["Data_Point"].astype(float),
            equal_nan=True,
        )),
        "all_original_measurements_preserved": all(measurement_equivalence.values()),
        "original_measurement_checks": measurement_equivalence,
        "phase_values_valid": set(df["Phase"].unique()).issubset(allowed_phases),
        "phase_ids_consecutive": bool(np.array_equal(phase_ids, expected_phase_ids)),
        "one_phase_per_phase_id": bool(df.groupby("Phase_ID")["Phase"].nunique().le(1).all()),
        "phase_transition_count_matches_segments": int(df["Phase_Transition"].sum()) == df["Phase_ID"].nunique(),
        "cycle_index_preserved": measurement_equivalence["Cycle_Index"],
        "temperature_missing_not_filled": int(df["Temperature(℃)"].isna().sum())
        == int(pd.to_numeric(raw["Temperature(℃)"], errors="coerce").isna().sum()),
    }
    failed = [name for name, value in validations.items() if name != "original_measurement_checks" and value is not True]
    if failed:
        raise RuntimeError(f"{source.name}: post-transformation validation failed: {failed}")

    phase_counts = {str(key): int(value) for key, value in df["Phase"].value_counts().items()}
    flag_columns = [column for column in df.columns if column.endswith("_Flag")]
    summary = {
        "configuration_sha256": CONFIG_SHA256,
        "configuration": CONFIG,
        "source_file": source.name,
        "input_rows": original_rows,
        "output_rows": len(df),
        "conversion_report": conversions,
        "cycle_index_min": int(df["Cycle_Index"].min()) if df["Cycle_Index"].notna().any() else None,
        "cycle_index_max": int(df["Cycle_Index"].max()) if df["Cycle_Index"].notna().any() else None,
        "unique_cycles": int(df["Cycle_Index"].nunique()),
        "phase_counts": phase_counts,
        "phase_segments": int(df["Phase_ID"].nunique()),
        "flag_counts": {column: int(df[column].sum()) for column in flag_columns},
        "sign_convention_evidence": sign_convention_evidence(df),
        "validations": validations,
    }
    return df, summary


def build_markdown(summaries, output_paths):
    lines = [
        "# Silver Transformation Report", "",
        "The two cleaned battery datasets were transformed without sorting, deleting, interpolating, clipping, or rounding source measurements.", "",
        "## Configuration", "",
        f"- Rest-current deadband: ±{REST_THRESHOLD_A} A",
        f"- Sampling-gap threshold: {SAMPLING_GAP_SECONDS} s using `Test_Time(s)`",
        f"- Charge-current screening limit: {MAX_CHARGE_CURRENT_A} A",
        f"- Discharge-current screening limit: {MAX_DISCHARGE_CURRENT_A} A",
        f"- Voltage screening window: {VOLTAGE_MIN_V}–{VOLTAGE_MAX_V} V", "",
    ]
    for summary, output in zip(summaries, output_paths):
        sign = summary["sign_convention_evidence"]
        lines += [
            f"## {summary['source_file']}", "",
            f"- Rows in / out: **{summary['input_rows']:,} / {summary['output_rows']:,}**",
            f"- Cycle range: **{summary['cycle_index_min']}–{summary['cycle_index_max']}** ({summary['unique_cycles']} observed)",
            f"- Phase segments: **{summary['phase_segments']:,}**",
            f"- Output: `{output.name}`", "",
            "### Phase rows", "", "| Phase | Rows |", "|---|---:|",
        ]
        for phase in ["CHARGE", "DISCHARGE", "REST", "UNKNOWN"]:
            lines.append(f"| {phase} | {summary['phase_counts'].get(phase, 0):,} |")
        lines += ["", "### Quality flags", "", "| Flag | Rows |", "|---|---:|"]
        for flag, count in summary["flag_counts"].items():
            lines.append(f"| {flag} | {count:,} |")
        lines += [
            "", "### Current-sign evidence", "",
            f"- Positive-current-as-charge supported: **{sign['positive_current_is_charge_supported']}**",
            f"- Median charge-segment voltage change: **{sign['charge']['median_segment_voltage_change_v']} V**",
            f"- Median discharge-segment voltage change: **{sign['discharge']['median_segment_voltage_change_v']} V**", "",
            "### Validation", "", "| Check | Result |", "|---|---|",
        ]
        for check, result in summary["validations"].items():
            if check != "original_measurement_checks":
                lines.append(f"| {check} | {'PASS' if result else 'FAIL'} |")
        lines.append("")
    lines += [
        "## Assumptions and cautions", "",
        "- Positive current is classified as charge and negative current as discharge, with voltage-direction evidence reported above.",
        "- Repeated and reversed timestamps are retained. `Test_Time(s)` remains the primary elapsed-time axis.",
        "- Missing temperature remains null and is explicitly flagged; no temperature values were invented.",
        "- Plausibility flags are screening indicators, not automatic deletion criteria.", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = args.inputs or DEFAULT_INPUTS
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    prepared, summaries = [], []
    for path in inputs:
        frame, summary = transform(pd.read_parquet(path), path)
        prepared.append((path, frame))
        summaries.append(summary)

    schemas = [list(frame.columns) for _, frame in prepared]
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise RuntimeError("Silver output schemas are inconsistent between cells")

    args.out.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for path, frame in prepared:
        name = path.name.replace("_clean.parquet", "_silver.parquet")
        output_path = args.out / name
        frame.to_parquet(output_path, index=False, compression="snappy")
        output_paths.append(output_path)
        print(f"Created {output_path} ({len(frame):,} rows)")

    report_path = args.out / "silver_transformation_report.md"
    json_path = args.out / "silver_transformation_summary.json"
    report_path.write_text(build_markdown(summaries, output_paths), encoding="utf-8")
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Created {report_path}")
    print(f"Created {json_path}")


if __name__ == "__main__":
    main()
