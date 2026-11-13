#!/usr/bin/env python3
"""Battery dataset quality audit for one or more Parquet files.

Run:
    python battery_data_audit_parquet.py file1.parquet file2.parquet --out audit_output

The input files are never modified. Suspicious rows are flagged, not deleted.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Data_Point", "Test_Time(s)", "Current(A)", "Capacity(Ah)",
    "Voltage(V)", "Energy(Wh)", "Temperature(℃)", "Date_Time", "Cycle_Index",
]
LIMITS = {
    "voltage_min": 0.0, "voltage_max": 5.0, "current_abs_max": 30.0,
    "capacity_min": 0.0, "capacity_max": 10.0, "energy_min": 0.0,
    "energy_max": 100.0, "temperature_min": -40.0, "temperature_max": 100.0,
}
REST_THRESHOLD_A = 0.05
SAMPLING_GAP_SECONDS = 5.0
SENTINELS = ["", "-", "--", "NA", "N/A", "null", "None"]
ALIASES = {
    "Data_Point": ["Data_Point", "Data Point", "DataPoint"],
    "Test_Time(s)": ["Test_Time(s)", "Test Time(s)", "Test_Time", "Test Time"],
    "Current(A)": ["Current(A)", "Current (A)", "Current_A", "Current"],
    "Capacity(Ah)": ["Capacity(Ah)", "Capacity (Ah)", "Capacity_Ah", "Capacity"],
    "Voltage(V)": ["Voltage(V)", "Voltage (V)", "Voltage_V", "Voltage"],
    "Energy(Wh)": ["Energy(Wh)", "Energy (Wh)", "Energy_Wh", "Energy"],
    "Temperature(℃)": ["Temperature(℃)", "Temperature(°C)", "Temperature(C)", "Temperature"],
    "Date_Time": ["Date_Time", "Date Time", "DateTime", "Timestamp"],
    "Cycle_Index": ["Cycle_Index", "Cycle Index", "CycleIndex", "Cycle"],
}


def normalize_header(name):
    return re.sub(r"\s+", " ", str(name).strip().replace("º", "°").replace("（", "(").replace("）", ")"))


def standardize_columns(df):
    normalized = {normalize_header(column): column for column in df.columns}
    mapping, missing = {}, []
    for standard, aliases in ALIASES.items():
        found = next((normalized[normalize_header(alias)] for alias in aliases if normalize_header(alias) in normalized), None)
        if found is None:
            missing.append(standard)
        else:
            mapping[standard] = found
    if not missing:
        df = df.rename(columns={original: standard for standard, original in mapping.items()})
    return df, mapping, missing


def add_flag(df, mask, name):
    if name not in df:
        df[name] = False
    df.loc[mask.fillna(False).astype(bool), name] = True


def convert_numeric(df):
    report = {}
    for column in [c for c in REQUIRED_COLUMNS if c != "Date_Time"]:
        original = df[column].copy()
        cleaned = original.astype("string").str.strip().replace(SENTINELS, pd.NA)
        converted = pd.to_numeric(cleaned, errors="coerce")
        report[column] = {
            "conversion_failures": int((original.notna() & converted.isna()).sum()),
            "missing_after_conversion": int(converted.isna().sum()),
        }
        df[column] = converted
    return df, report


def audit_dataset(raw, dataset_name):
    df, mapping, missing = standardize_columns(raw.copy())
    summary = {
        "dataset": dataset_name, "input_rows": len(raw), "input_columns": list(raw.columns),
        "missing_required_columns": missing, "column_mapping": mapping, "warnings": [],
    }
    if missing:
        summary["warnings"].append("Required columns missing; audit stopped for this dataset.")
        return df, summary, pd.DataFrame(), pd.DataFrame()

    df.insert(0, "Source_Dataset", dataset_name)
    df.insert(1, "Source_Row", np.arange(len(df)) + 1)
    df, summary["numeric_conversion"] = convert_numeric(df)

    original_timestamp = df["Date_Time"].copy()
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
    test_diff = df["Test_Time(s)"].diff()
    timestamp_diff = df["Date_Time"].diff().dt.total_seconds()
    add_flag(df, test_diff < 0, "flag_test_time_decrease")
    add_flag(df, timestamp_diff < 0, "flag_timestamp_reversal")
    add_flag(df, test_diff > SAMPLING_GAP_SECONDS, "flag_sampling_gap")
    add_flag(df, timestamp_diff > SAMPLING_GAP_SECONDS, "flag_timestamp_gap")

    valid = df["Date_Time"].notna() & df["Test_Time(s)"].notna()
    df["Timestamp_TestTime_Difference_s"] = np.nan
    if valid.any():
        timestamp_elapsed = (df.loc[valid, "Date_Time"] - df.loc[valid, "Date_Time"].iloc[0]).dt.total_seconds()
        test_elapsed = df.loc[valid, "Test_Time(s)"] - df.loc[valid, "Test_Time(s)"].iloc[0]
        df.loc[valid, "Timestamp_TestTime_Difference_s"] = timestamp_elapsed.to_numpy() - test_elapsed.to_numpy()
    add_flag(df, df["Timestamp_TestTime_Difference_s"].abs() > 2, "flag_timestamp_testtime_mismatch")
    summary["timestamp_checks"] = {
        "timestamp_conversion_failures": int((original_timestamp.notna() & df["Date_Time"].isna()).sum()),
        "test_time_decreases": int((test_diff < 0).sum()),
        "timestamp_reversals": int((timestamp_diff < 0).sum()),
        "test_time_max_gap": float(test_diff.max()) if test_diff.notna().any() else None,
        "timestamp_max_gap": float(timestamp_diff.max()) if timestamp_diff.notna().any() else None,
    }

    critical = [c for c in REQUIRED_COLUMNS if c != "Temperature(℃)"]
    summary["missing_values"] = {column: int(df[column].isna().sum()) for column in REQUIRED_COLUMNS}
    add_flag(df, df[critical].isna().any(axis=1), "flag_critical_missing")
    add_flag(df, df["Temperature(℃)"].isna(), "flag_temperature_missing")
    if df["Temperature(℃)"].isna().all():
        summary["warnings"].append("Temperature unavailable in entire dataset.")

    duplicate_masks = {
        "exact_duplicate_rows": df[REQUIRED_COLUMNS].duplicated(keep=False),
        "duplicate_datapoint_rows": df["Data_Point"].duplicated(keep=False),
        "duplicate_timestamp_rows": df["Date_Time"].duplicated(keep=False),
        "duplicate_cycle_time_rows": df.duplicated(["Cycle_Index", "Test_Time(s)"], keep=False),
    }
    for metric, mask in duplicate_masks.items():
        add_flag(df, mask, "flag_" + metric.removesuffix("_rows"))
    summary["duplicates"] = {metric: int(mask.sum()) for metric, mask in duplicate_masks.items()}

    # Investigate repeated Date_Time values without deleting or merging records.
    duplicate_timestamp_mask = duplicate_masks["duplicate_timestamp_rows"] & df["Date_Time"].notna()
    duplicate_timestamp_rows = df.loc[duplicate_timestamp_mask].copy()
    duplicate_groups = duplicate_timestamp_rows.groupby("Date_Time", sort=False, dropna=True)
    test_time_nunique = duplicate_groups["Test_Time(s)"].nunique(dropna=False)
    timestamp_test_time_pair_mask = (
        df.duplicated(["Date_Time", "Test_Time(s)"], keep=False) & df["Date_Time"].notna()
    )
    measurement_differences = {}
    for column in ["Current(A)", "Voltage(V)", "Capacity(Ah)", "Energy(Wh)"]:
        value_counts = duplicate_groups[column].nunique(dropna=False)
        measurement_differences[column] = {
            "groups_with_different_values": int((value_counts > 1).sum()),
            "groups_with_identical_values": int((value_counts <= 1).sum()),
        }
    duplicate_rest = duplicate_timestamp_rows["Current(A)"].abs() <= REST_THRESHOLD_A
    duplicate_positive = duplicate_timestamp_rows["Current(A)"] > REST_THRESHOLD_A
    duplicate_negative = duplicate_timestamp_rows["Current(A)"] < -REST_THRESHOLD_A
    valid_timestamp = df["Date_Time"].notna()
    whole_second = (
        valid_timestamp
        & df["Date_Time"].dt.microsecond.eq(0)
        & df["Date_Time"].dt.nanosecond.eq(0)
    )
    summary["duplicate_timestamp_diagnostics"] = {
        "duplicate_timestamp_groups": int(duplicate_groups.ngroups),
        "duplicate_timestamp_rows": int(duplicate_timestamp_mask.sum()),
        "groups_where_test_time_is_identical": int((test_time_nunique <= 1).sum()),
        "groups_where_test_time_differs": int((test_time_nunique > 1).sum()),
        "rows_with_duplicated_timestamp_and_test_time_pair": int(timestamp_test_time_pair_mask.sum()),
        "measurement_differences_by_timestamp_group": measurement_differences,
        "duplicate_rows_during_rest": int(duplicate_rest.sum()),
        "duplicate_rows_during_positive_current": int(duplicate_positive.sum()),
        "duplicate_rows_during_negative_current": int(duplicate_negative.sum()),
        "duplicate_rows_rest_percentage": float(100 * duplicate_rest.mean()) if len(duplicate_rest) else None,
        "duplicates_primarily_during_rest": bool(duplicate_rest.mean() > 0.5) if len(duplicate_rest) else None,
        "valid_timestamps": int(valid_timestamp.sum()),
        "timestamps_on_whole_seconds": int(whole_second.sum()),
        "whole_second_percentage": float(100 * whole_second.sum() / valid_timestamp.sum()) if valid_timestamp.any() else None,
        "all_timestamps_rounded_to_whole_seconds": bool(whole_second.sum() == valid_timestamp.sum()) if valid_timestamp.any() else None,
    }

    plausibility = {
        "voltage_out_of_range": ~df["Voltage(V)"].between(LIMITS["voltage_min"], LIMITS["voltage_max"]),
        "current_out_of_range": df["Current(A)"].abs() > LIMITS["current_abs_max"],
        "capacity_negative": df["Capacity(Ah)"] < LIMITS["capacity_min"],
        "capacity_too_high": df["Capacity(Ah)"] > LIMITS["capacity_max"],
        "energy_negative": df["Energy(Wh)"] < LIMITS["energy_min"],
        "energy_too_high": df["Energy(Wh)"] > LIMITS["energy_max"],
        "temperature_out_of_range": ~df["Temperature(℃)"].between(LIMITS["temperature_min"], LIMITS["temperature_max"]),
    }
    for name, mask in plausibility.items():
        add_flag(df, mask, f"flag_{name}")
    summary["plausibility_checks"] = {name: int(mask.fillna(False).sum()) for name, mask in plausibility.items()}

    current = df["Current(A)"]
    positive, negative, rest = current > REST_THRESHOLD_A, current < -REST_THRESHOLD_A, current.abs() <= REST_THRESHOLD_A
    df["Current_State"] = np.select([positive, negative], ["positive_current", "negative_current"], default="rest")
    df["Operating_Phase"] = np.select([positive, negative], ["charge", "discharge"], default="rest")
    df["Sign_Convention_Verified"] = False
    summary["current_sign"] = {
        "positive_current_rows": int(positive.sum()), "negative_current_rows": int(negative.sum()),
        "rest_rows": int(rest.sum()), "positive_assumed_charge": True,
        "negative_assumed_discharge": True, "verified": False,
    }

    cycle_values = sorted(df["Cycle_Index"].dropna().unique())
    integer_cycles = [int(cycle) for cycle in cycle_values if float(cycle).is_integer()]
    missing_cycles = sorted(set(range(min(integer_cycles), max(integer_cycles) + 1)) - set(integer_cycles)) if integer_cycles else []
    summary["cycle_checks"] = {
        "minimum_cycle": min(integer_cycles) if integer_cycles else None,
        "maximum_cycle": max(integer_cycles) if integer_cycles else None,
        "unique_cycles": len(cycle_values), "missing_cycle_indices": missing_cycles,
    }

    cycle_rows = []
    for cycle, group in df.groupby("Cycle_Index", dropna=True, sort=True):
        charge = group[group["Current(A)"] > REST_THRESHOLD_A]
        discharge = group[group["Current(A)"] < -REST_THRESHOLD_A]
        rest_group = group[group["Current(A)"].abs() <= REST_THRESHOLD_A]
        cycle_rows.append({
            "Cycle_Index": cycle, "Rows": len(group), "Charge_Rows": len(charge),
            "Discharge_Rows": len(discharge), "Rest_Rows": len(rest_group),
            "Has_Charge": len(charge) > 0, "Has_Discharge": len(discharge) > 0,
            "Complete_Charge_Discharge": len(charge) > 0 and len(discharge) > 0,
            "Capacity_Min_Ah": group["Capacity(Ah)"].min(), "Capacity_Max_Ah": group["Capacity(Ah)"].max(),
            "Voltage_Min_V": group["Voltage(V)"].min(), "Voltage_Max_V": group["Voltage(V)"].max(),
            "Energy_Min_Wh": group["Energy(Wh)"].min(), "Energy_Max_Wh": group["Energy(Wh)"].max(),
            "First_Test_Time": group["Test_Time(s)"].min(), "Last_Test_Time": group["Test_Time(s)"].max(),
        })
    cycle_summary = pd.DataFrame(cycle_rows)

    capacity_reset = df.groupby("Cycle_Index")["Capacity(Ah)"].diff() < -0.05
    energy_reset = df.groupby("Cycle_Index")["Energy(Wh)"].diff() < -0.5
    add_flag(df, capacity_reset, "flag_capacity_reset")
    add_flag(df, energy_reset, "flag_energy_reset")
    add_flag(df, (current.abs() > REST_THRESHOLD_A) & df["Voltage(V)"].isna(), "flag_missing_voltage_during_current")
    summary["reset_checks"] = {"capacity_reset_rows": int(capacity_reset.sum()), "energy_reset_rows": int(energy_reset.sum())}

    flag_columns = [column for column in df if column.startswith("flag_")]
    summary["flag_counts"] = {column: int(df[column].fillna(False).sum()) for column in flag_columns}
    summary["output_rows"] = len(df)
    summary["warnings"] += [
        "Current sign convention is provisional.", "Reset flags do not prove invalid data.",
        "Incomplete cycles must not be used for complete-cycle efficiency calculations.",
    ]
    flagged = df.loc[df[flag_columns].fillna(False).any(axis=1)].copy()
    return df, summary, cycle_summary, flagged


def markdown_report(summaries, generated_files):
    lines = ["# Battery Dataset Quality Audit", "", "## Scope and method", "",
             "The two Parquet datasets were audited independently. No input rows were deleted or modified. "
             "All screening limits are broad audit limits from the supplied program, not verified manufacturer limits.", ""]
    for summary in summaries:
        name = summary["dataset"]
        lines += [f"## {name}", "", f"- Rows audited: **{summary['input_rows']:,}**",
                  f"- Rows carrying at least one audit flag: **{generated_files[name]['flagged_rows']:,}**",
                  f"- Required columns missing: **{len(summary['missing_required_columns'])}**"]
        if summary["missing_required_columns"]:
            lines += [f"- Missing fields: `{', '.join(summary['missing_required_columns'])}`", ""]
            continue
        missing = summary["missing_values"]
        dup = summary["duplicates"]
        time = summary["timestamp_checks"]
        cycles = summary["cycle_checks"]
        sign = summary["current_sign"]
        reset = summary["reset_checks"]
        diagnostics = summary["duplicate_timestamp_diagnostics"]
        complete = generated_files[name]["complete_cycles"]
        incomplete = generated_files[name]["incomplete_cycles"]
        lines += [
            f"- Critical-field missing values: **{sum(missing[c] for c in REQUIRED_COLUMNS if c != 'Temperature(℃)'):,}**",
            f"- Missing temperature readings: **{missing['Temperature(℃)']:,}**",
            f"- Exact duplicate rows: **{dup['exact_duplicate_rows']:,}**",
            f"- Duplicate timestamps: **{dup['duplicate_timestamp_rows']:,}**",
            f"- Timestamp reversals: **{time['timestamp_reversals']:,}**",
            f"- Test-time decreases: **{time['test_time_decreases']:,}**",
            f"- Cycle-index range: **{cycles['minimum_cycle']}–{cycles['maximum_cycle']}** ({cycles['unique_cycles']} observed)",
            f"- Cycles containing both charge and discharge rows: **{complete:,}**",
            f"- Cycles lacking charge or discharge rows: **{incomplete:,}**",
            f"- Positive / negative / rest rows: **{sign['positive_current_rows']:,} / {sign['negative_current_rows']:,} / {sign['rest_rows']:,}**",
            f"- Capacity-reset flags: **{reset['capacity_reset_rows']:,}**",
            f"- Energy-reset flags: **{reset['energy_reset_rows']:,}**", "",
            "### Duplicate-timestamp diagnostics", "",
            f"- Repeated timestamp groups: **{diagnostics['duplicate_timestamp_groups']:,}**",
            f"- Rows sharing both `Date_Time` and `Test_Time(s)`: **{diagnostics['rows_with_duplicated_timestamp_and_test_time_pair']:,}**",
            f"- Groups with identical / different `Test_Time(s)`: **{diagnostics['groups_where_test_time_is_identical']:,} / {diagnostics['groups_where_test_time_differs']:,}**",
            f"- Repeated timestamp rows during rest / positive / negative current: **{diagnostics['duplicate_rows_during_rest']:,} / {diagnostics['duplicate_rows_during_positive_current']:,} / {diagnostics['duplicate_rows_during_negative_current']:,}**",
            f"- Rest share of repeated timestamp rows: **{diagnostics['duplicate_rows_rest_percentage']:.2f}%**",
            f"- Duplicates occur primarily during rest: **{diagnostics['duplicates_primarily_during_rest']}**",
            f"- Exact whole-second timestamps: **{diagnostics['timestamps_on_whole_seconds']:,} / {diagnostics['valid_timestamps']:,} ({diagnostics['whole_second_percentage']:.2f}%)**",
            f"- All timestamps rounded to whole seconds: **{diagnostics['all_timestamps_rounded_to_whole_seconds']}**", "",
            "| Measurement | Groups with different values | Groups with identical values |",
            "|---|---:|---:|",
        ]
        for measurement, result in diagnostics["measurement_differences_by_timestamp_group"].items():
            lines.append(
                f"| {measurement} | {result['groups_with_different_values']:,} | "
                f"{result['groups_with_identical_values']:,} |"
            )
        lines += ["",
            "### Plausibility flags", "",
            "| Check | Flagged rows |", "|---|---:|",
        ]
        for check, count in summary["plausibility_checks"].items():
            lines.append(f"| {check.replace('_', ' ')} | {count:,} |")
        lines += ["", "### Interpretation", "",
                  "- Every row carries a temperature-missing flag because the source temperature column contains no measurements. This does not, by itself, invalidate the electrical measurements.",
                  "- Positive current is treated provisionally as charge and negative current as discharge; this must be confirmed from voltage trends and the test protocol.",
                  "- Capacity and energy reset flags describe counter behavior and are not grounds for deleting rows.",
                  "- Cycle completeness here only means both current polarities occur within a Cycle_Index; boundary voltage, duration, and sequencing checks are still required before efficiency calculations.", ""]
    lines += ["## Generated audit files", ""]
    for name, files in generated_files.items():
        lines += [f"### {name}", "", f"- Audited rows: `{files['audited']}`",
                  f"- Flagged rows: `{files['flagged']}`", f"- Cycle summary: `{files['cycles']}`", ""]
    lines += ["The machine-readable summary is stored in `data_quality_summary.json`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="One or more Parquet input files")
    parser.add_argument("--out", default="audit_output", help="Output directory")
    args = parser.parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, generated = [], {}
    for input_name in args.inputs:
        path = Path(input_name)
        if not path.exists():
            raise FileNotFoundError(path)
        raw = pd.read_parquet(path)
        audited, summary, cycles, flagged = audit_dataset(raw, path.stem)
        key = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem)
        audited_path = output_dir / f"audited_{key}.parquet"
        flagged_path = output_dir / f"flagged_rows_{key}.parquet"
        cycles_path = output_dir / f"cycle_summary_{key}.csv"
        audited.to_parquet(audited_path, index=False, compression="snappy")
        flagged.to_parquet(flagged_path, index=False, compression="snappy")
        cycles.to_csv(cycles_path, index=False)
        complete = int(cycles["Complete_Charge_Discharge"].sum()) if not cycles.empty else 0
        generated[path.stem] = {
            "audited": audited_path.name, "flagged": flagged_path.name, "cycles": cycles_path.name,
            "complete_cycles": complete, "incomplete_cycles": len(cycles) - complete,
            "flagged_rows": len(flagged),
        }
        summaries.append(summary)
        print(f"{path.name}: rows={len(audited):,}, flagged={len(flagged):,}, cycles={len(cycles):,}")
    with (output_dir / "data_quality_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False, default=str)
    (output_dir / "data_quality_report.md").write_text(markdown_report(summaries, generated), encoding="utf-8")
    print(f"Report: {(output_dir / 'data_quality_report.md').resolve()}")


if __name__ == "__main__":
    main()
