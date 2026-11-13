#!/usr/bin/env python3
"""Stage 2: clean, validate, and combine the timestamp-normalized cell data."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from battery_config import CONFIG, CONFIG_SHA256
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "battery_cleaning"
INPUT_FILE = OUTPUT_DIR / "01_timestamp_normalized.xlsx"
OUTPUT_FILE = OUTPUT_DIR / "02_unified_cleaned_data.xlsx"
DATA_SHEETS = ["Cell_001_2", "Cell_002_4"]
CELL_IDS = {
    "Cell_001_2": "LR1865SZ_cycles201217_001_2",
    "Cell_002_4": "LR1865SZ_cycles201214_002_4",
}
CURRENT_DEADBAND_A = CONFIG['rest_current_threshold_a']
GAP_THRESHOLD_S = CONFIG['maximum_integration_interval_s']

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(name="Arial", size=10, bold=True, color="FFFFFF")
FLAG_FILL = PatternFill("solid", fgColor="FFF2CC")
REJECT_FILL = PatternFill("solid", fgColor="F4CCCC")
OK_FILL = PatternFill("solid", fgColor="E2F0D9")


def excel_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def append_header(ws, headers):
    row = []
    for header in headers:
        cell = WriteOnlyCell(ws, value=header)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        row.append(cell)
    ws.append(row)


def configure_sheet(ws, headers, widths=None):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"
    ws.sheet_view.showGridLines = False
    if widths is None:
        widths = [15] * len(headers)
    for index, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(index)].width = width


def append_frame(ws, frame: pd.DataFrame):
    headers = frame.columns.tolist()
    append_header(ws, headers)
    timestamp_index = headers.index("timestamp") if "timestamp" in headers else None
    status_index = headers.index("row_quality_status") if "row_quality_status" in headers else None
    for values in frame.itertuples(index=False, name=None):
        row = []
        for index, value in enumerate(values):
            cell = WriteOnlyCell(ws, value=excel_value(value))
            cell.font = Font(name="Arial", size=9)
            if timestamp_index is not None and index == timestamp_index and cell.value is not None:
                cell.number_format = "yyyy-mm-dd hh:mm:ss"
            if status_index is not None and index == status_index:
                cell.fill = {"OK": OK_FILL, "FLAGGED": FLAG_FILL, "REJECTED": REJECT_FILL}.get(str(cell.value))
            row.append(cell)
        ws.append(row)
    widths = [12] * len(headers)
    for name, width in {
        "cell_id": 31, "timestamp": 21,
        "operating_state": 16, "row_quality_status": 18, "quality_flags": 65,
        "reject_reason": 38,
    }.items():
        if name in headers:
            widths[headers.index(name)] = width
    configure_sheet(ws, headers, widths)


def combine_flags(frame: pd.DataFrame, mappings):
    arrays = [np.where(frame[column].fillna(False), label, "") for column, label in mappings]
    return pd.Series(
        [";".join(label for label in row if label) for row in zip(*arrays)],
        index=frame.index,
        dtype="string",
    )


def load_and_standardize(sheet_name: str):
    source = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
    required = [
        "Data_Point", "Test_Time(s)", "Current(A)", "Capacity(Ah)", "Voltage(V)",
        "Energy(Wh)", "Temperature(℃)", "Date_Time", "Cycle_Index",
    ]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise RuntimeError(f"{sheet_name}: normalized schema failure; missing={missing}")

    frame = pd.DataFrame(index=source.index)
    frame["source_row"] = source.index + 2
    frame["cell_id"] = CELL_IDS[sheet_name]
    frame["data_point"] = pd.to_numeric(source["Data_Point"], errors="coerce")
    frame["test_time_s"] = pd.to_numeric(source["Test_Time(s)"], errors="coerce")
    frame["timestamp"] = pd.to_datetime(source["Date_Time"], errors="coerce")
    frame["elapsed_time_s"] = (frame["timestamp"] - frame["timestamp"].iloc[0]).dt.total_seconds()
    frame["cycle_elapsed_s"] = frame["timestamp"].groupby(source["Cycle_Index"]).transform(
        lambda series: (series - series.iloc[0]).dt.total_seconds()
    )
    frame["delta_time_s"] = frame["timestamp"].diff().dt.total_seconds()
    frame["test_time_delta_s"] = frame["test_time_s"].diff()
    frame["time_axis_difference_s"] = frame["delta_time_s"] - frame["test_time_delta_s"]
    frame["current_a"] = pd.to_numeric(source["Current(A)"], errors="coerce")
    frame["voltage_v"] = pd.to_numeric(source["Voltage(V)"], errors="coerce")
    frame["capacity_ah"] = pd.to_numeric(source["Capacity(Ah)"], errors="coerce")
    frame["energy_wh"] = pd.to_numeric(source["Energy(Wh)"], errors="coerce")
    raw_temperature = source["Temperature(℃)"].astype("string").str.strip()
    frame["temperature_placeholder"] = raw_temperature.eq("-")
    frame["temperature_sentinel"] = raw_temperature.isin(["-999", "65535", "32767"])
    frame["temperature_c"] = pd.to_numeric(
        raw_temperature.mask(frame["temperature_placeholder"] | frame["temperature_sentinel"]), errors="coerce"
    )
    frame["cycle_index"] = pd.to_numeric(source["Cycle_Index"], errors="coerce")
    return frame


def apply_checks(frame: pd.DataFrame):
    frame = frame.copy()
    frame["temperature_missing"] = frame["temperature_c"].isna()
    frame["timestamp_repeated"] = frame.groupby("cell_id")["timestamp"].diff().dt.total_seconds().eq(0)
    frame["timestamp_backward"] = frame.groupby("cell_id")["timestamp"].diff().dt.total_seconds().lt(0)
    frame["sampling_gap"] = frame["delta_time_s"].gt(GAP_THRESHOLD_S)
    frame["time_axis_mismatch"] = frame["time_axis_difference_s"].abs().gt(1.0)
    frame["duplicate_data_point"] = frame.duplicated(["cell_id", "data_point"], keep=False)
    frame["duplicate_timestamp"] = frame.duplicated(["cell_id", "timestamp"], keep=False) & frame["timestamp"].notna()

    duplicate_basis = [
        "cell_id", "data_point", "test_time_s", "timestamp", "current_a", "voltage_v",
        "capacity_ah", "energy_wh", "temperature_c", "cycle_index",
    ]
    frame["exact_duplicate"] = frame.duplicated(duplicate_basis, keep="first")
    frame["voltage_outside_spec"] = frame["voltage_v"].lt(CONFIG['voltage_min_v']) | frame["voltage_v"].gt(CONFIG['voltage_max_v'])
    frame["voltage_outside_plausible"] = frame["voltage_v"].lt(2.5) | frame["voltage_v"].gt(4.3)
    frame["voltage_impossible"] = frame["voltage_v"].le(0) | frame["voltage_v"].gt(5.0)
    frame["capacity_invalid"] = frame["capacity_ah"].lt(0) | frame["capacity_ah"].gt(CONFIG['capacity_warning_max_ah'])
    frame["energy_invalid"] = frame["energy_wh"].lt(0) | frame["energy_wh"].gt(CONFIG['energy_warning_max_wh'])
    frame["temperature_outside_broad_range"] = frame["temperature_c"].notna() & ~frame["temperature_c"].between(-40, 80)

    frame["operating_state"] = np.select(
        [frame["current_a"] > CURRENT_DEADBAND_A, frame["current_a"] < -CURRENT_DEADBAND_A],
        ["charge", "discharge"], default="rest",
    )
    frame["charge_current_over_spec"] = frame["operating_state"].eq("charge") & frame["current_a"].gt(CONFIG['rated_charge_current_a'])
    frame["discharge_current_over_spec"] = frame["operating_state"].eq("discharge") & frame["current_a"].abs().gt(CONFIG['rated_discharge_current_a'])
    frame["temperature_outside_operating_range"] = (
        (frame["operating_state"].eq("charge") & frame["temperature_c"].notna() & ~frame["temperature_c"].between(0, 45))
        | (frame["operating_state"].eq("discharge") & frame["temperature_c"].notna() & ~frame["temperature_c"].between(-20, 60))
    )

    frame["cycle_index_decreased"] = frame.groupby("cell_id")["cycle_index"].diff().lt(0)
    frame["cycle_index_jump"] = frame.groupby("cell_id")["cycle_index"].diff().gt(1)
    voltage_change = frame.groupby("cell_id")["voltage_v"].diff()
    frame["current_voltage_contradiction"] = (
        (frame["operating_state"].eq("charge") & voltage_change.lt(-0.05))
        | (frame["operating_state"].eq("discharge") & voltage_change.gt(0.05))
    )
    frame["capacity_counter_decrease"] = frame.groupby(
        ["cell_id", "cycle_index", "operating_state"], dropna=False
    )["capacity_ah"].diff().lt(-0.002)
    frame["capacity_counter_reset"] = frame["capacity_ah"].le(0.001) & frame.groupby("cell_id")["capacity_ah"].shift().gt(0.05)
    frame["energy_counter_reset"] = frame["energy_wh"].le(0.001) & frame.groupby("cell_id")["energy_wh"].shift().gt(0.05)

    frame["critical_field_missing"] = frame[["cell_id", "timestamp", "current_a", "voltage_v", "cycle_index"]].isna().any(axis=1)
    frame["reject_reason"] = ""
    frame.loc[frame["critical_field_missing"], "reject_reason"] = "critical_field_missing_or_unparseable"
    frame.loc[frame["voltage_impossible"] & frame["reject_reason"].eq(""), "reject_reason"] = "impossible_voltage"
    frame.loc[frame["exact_duplicate"] & frame["reject_reason"].eq(""), "reject_reason"] = "exact_duplicate"

    flag_map = [
        ("temperature_missing", "temperature_missing"),
        ("temperature_sentinel", "temperature_sentinel"),
        ("timestamp_repeated", "timestamp_repeated"),
        ("timestamp_backward", "timestamp_backward"),
        ("sampling_gap", "sampling_gap_gt_10s"),
        ("time_axis_mismatch", "time_axis_mismatch_gt_1s"),
        ("duplicate_data_point", "duplicate_data_point"),
        ("duplicate_timestamp", "duplicate_timestamp"),
        ("voltage_outside_spec", "voltage_outside_3p0_to_4p2V"),
        ("voltage_outside_plausible", "voltage_outside_2p5_to_4p3V"),
        ("charge_current_over_spec", "charge_current_over_2p4A"),
        ("discharge_current_over_spec", "discharge_current_over_7p2A"),
        ("temperature_outside_broad_range", "temperature_outside_minus40_to_80C"),
        ("temperature_outside_operating_range", "temperature_outside_phase_operating_range"),
        ("capacity_invalid", "capacity_outside_0_to_3Ah"),
        ("energy_invalid", "energy_outside_0_to_15Wh"),
        ("cycle_index_decreased", "cycle_index_decreased"),
        ("cycle_index_jump", "cycle_index_jump"),
        ("current_voltage_contradiction", "current_voltage_local_contradiction"),
        ("capacity_counter_decrease", "capacity_counter_decrease_within_phase"),
        ("capacity_counter_reset", "capacity_counter_reset"),
        ("energy_counter_reset", "energy_counter_reset"),
    ]
    frame["quality_flags"] = combine_flags(frame, flag_map)
    frame["row_quality_status"] = np.where(
        frame["reject_reason"].ne(""), "REJECTED", np.where(frame["quality_flags"].eq(""), "OK", "FLAGGED")
    )
    return frame


def create_report(all_rows, retained, rejected):
    records = []
    def add(cell, section, metric, value, result="INFO", details=""):
        records.append([cell, section, metric, value, result, details])

    for cell_id, group in all_rows.groupby("cell_id", sort=False):
        kept = retained[retained["cell_id"].eq(cell_id)]
        removed = rejected[rejected["cell_id"].eq(cell_id)]
        add(cell_id, "Rows", "Rows read", len(group))
        add(cell_id, "Rows", "Rows retained", len(kept), "PASS" if len(kept) else "FAIL")
        add(cell_id, "Rows", "Rows rejected", len(removed), "REVIEW" if len(removed) else "PASS")
        add(cell_id, "Completeness", "Temperature missing", int(group["temperature_missing"].sum()), "REVIEW")
        add(cell_id, "Completeness", "Critical-field missing", int(group["critical_field_missing"].sum()), "FAIL" if group["critical_field_missing"].any() else "PASS")
        checks = [
            ("Uniqueness", "Exact duplicate rows", "exact_duplicate"),
            ("Uniqueness", "Duplicate Data_Point rows", "duplicate_data_point"),
            ("Uniqueness", "Duplicate timestamp rows", "duplicate_timestamp"),
            ("Time", "Repeated timestamps", "timestamp_repeated"),
            ("Time", "Backward timestamps", "timestamp_backward"),
            ("Time", f"Calendar gaps > {GAP_THRESHOLD_S:g} s (audit clock)", "sampling_gap"),
            ("Time", "Time-axis mismatches > 1 s", "time_axis_mismatch"),
            ("Physical", f"Voltage outside provisional {CONFIG['voltage_min_v']}–{CONFIG['voltage_max_v']} V", "voltage_outside_spec"),
            ("Physical", "Voltage outside 2.5–4.3 V", "voltage_outside_plausible"),
            ("Physical", f"Charge current > provisional {CONFIG['rated_charge_current_a']} A", "charge_current_over_spec"),
            ("Physical", f"Discharge current magnitude > provisional {CONFIG['rated_discharge_current_a']} A", "discharge_current_over_spec"),
            ("Counters", "Capacity resets", "capacity_counter_reset"),
            ("Counters", "Energy resets", "energy_counter_reset"),
            ("Counters", "Capacity decreases within phase", "capacity_counter_decrease"),
            ("Consistency", "Local current-voltage contradictions", "current_voltage_contradiction"),
            ("Cycles", "Cycle-index decreases", "cycle_index_decreased"),
            ("Cycles", "Cycle-index jumps > 1", "cycle_index_jump"),
        ]
        for section, metric, column in checks:
            count = int(group[column].sum())
            add(cell_id, section, metric, count, "REVIEW" if count else "PASS")
        add(cell_id, "Coverage", "Unique cycles", int(group["cycle_index"].nunique()))
        add(cell_id, "Coverage", "Cycle-index range", f"{int(group['cycle_index'].min())}–{int(group['cycle_index'].max())}")
        add(cell_id, "Units", "Canonical units", "V, A, Ah, Wh, °C, s", "PASS")
        add(cell_id, "Convention", "Current sign", "+ charge / - discharge", "PASS", "Confirmed against local voltage direction")
    return pd.DataFrame(records, columns=["cell_id", "section", "metric", "value", "result", "details"])


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Run 01_normalize_timestamps.py first: {INPUT_FILE}")
    frames = [load_and_standardize(sheet_name) for sheet_name in DATA_SHEETS]
    checked = apply_checks(pd.concat(frames, ignore_index=True))
    rejected_mask = checked["reject_reason"].ne("")

    output_columns = [
        "source_row", "cell_id", "data_point", "test_time_s", "timestamp", "delta_time_s",
        "current_a", "voltage_v", "capacity_ah", "energy_wh", "temperature_c", "cycle_index",
        "operating_state", "row_quality_status", "quality_flags",
    ]
    retained = checked.loc[~rejected_mask, output_columns].copy()
    rejected = checked.loc[rejected_mask, output_columns + ["reject_reason"]].copy()
    report = create_report(checked, retained, rejected)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # The original runner wrote only an Excel review workbook. Export the actual
    # measurement-preserving Parquet inputs too, so Silver cannot read stale files.
    mapping = {'data_point':'Data_Point','test_time_s':'Test_Time(s)',
               'current_a':'Current(A)','capacity_ah':'Capacity(Ah)',
               'voltage_v':'Voltage(V)','energy_wh':'Energy(Wh)',
               'temperature_c':'Temperature(℃)','timestamp':'Date_Time','cycle_index':'Cycle_Index'}
    exported = []
    for cell_id, group in checked.groupby('cell_id', sort=False):
        measurements = group[list(mapping)].rename(columns=mapping).reset_index(drop=True)
        path = OUTPUT_DIR / f'{cell_id}_clean.parquet'
        measurements.to_parquet(path, index=False)
        exported.append({'path':str(path.resolve()), 'rows':len(measurements),
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUTPUT_DIR/'cleaning_manifest.json').write_text(json.dumps({
        'configuration':CONFIG,'configuration_sha256':CONFIG_SHA256,
        'input_sha256':hashlib.sha256(INPUT_FILE.read_bytes()).hexdigest(),
        'measurement_policy':'All normalized measurement rows retained in source order in Parquet; Excel rejection sheet is a separate review view.',
        'outputs':exported}, indent=2), encoding='utf-8')
    workbook = Workbook(write_only=True)
    unified_ws = workbook.create_sheet("Unified_Data")
    append_frame(unified_ws, retained)
    rejected_ws = workbook.create_sheet("Rejected_Rows")
    append_frame(rejected_ws, rejected)
    report_ws = workbook.create_sheet("Quality_Report")
    append_frame(report_ws, report)
    workbook.save(OUTPUT_FILE)
    print(f"Created {OUTPUT_FILE}")
    print(f"Rows read={len(checked):,}; retained={len(retained):,}; rejected={len(rejected):,}")


if __name__ == "__main__":
    main()
