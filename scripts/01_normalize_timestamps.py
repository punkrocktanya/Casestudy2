#!/usr/bin/env python3
"""Stage 1: preserve measurements and normalize the two time axes only."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "battery_cleaning"
OUTPUT_FILE = OUTPUT_DIR / "01_timestamp_normalized.xlsx"
SOURCES = [
    (ROOT / "LR1865SZ_cycles201217_001_2.xlsx", "Cell_001_2"),
    (ROOT / "LR1865SZ_cycles201214_002_4.xlsx", "Cell_002_4"),
]
REQUIRED = [
    "Data_Point", "Test_Time(s)", "Current(A)", "Capacity(Ah)",
    "Voltage(V)", "Energy(Wh)", "Temperature(℃)", "Date_Time", "Cycle_Index",
]
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(name="Arial", size=10, bold=True, color="FFFFFF")


def time_to_seconds(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, time):
        return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    day_seconds = 0.0
    if "-" in text:
        days, text = text.split("-", 1)
        day_seconds = float(days) * 86400
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported Test_Time(s) value: {value!r}")
    hours, minutes, seconds = parts
    return day_seconds + float(hours) * 3600 + float(minutes) * 60 + float(seconds)


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


def configure_sheet(ws, headers, widths):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"
    ws.sheet_view.showGridLines = False
    for index, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(index)].width = width


def normalize_source(path: Path, sheet_output_name: str):
    xls = pd.ExcelFile(path)
    if len(xls.sheet_names) != 1:
        raise RuntimeError(f"{path.name}: expected one source sheet, found {xls.sheet_names}")
    source_sheet = xls.sheet_names[0]
    raw = pd.read_excel(path, sheet_name=source_sheet)
    missing = [column for column in REQUIRED if column not in raw.columns]
    duplicate_columns = raw.columns[raw.columns.duplicated()].tolist()
    if missing or duplicate_columns:
        raise RuntimeError(f"{path.name}: schema failure; missing={missing}, duplicate={duplicate_columns}")

    parsed_timestamp = pd.to_datetime(raw["Date_Time"], errors="coerce")
    try:
        test_seconds = raw["Test_Time(s)"].map(time_to_seconds)
    except Exception as exc:
        raise RuntimeError(f"{path.name}: failed to parse Test_Time(s): {exc}") from exc
    if parsed_timestamp.isna().any() or test_seconds.isna().any():
        raise RuntimeError(
            f"{path.name}: timestamp hard failure; Date_Time failures={int(parsed_timestamp.isna().sum())}, "
            f"Test_Time(s) failures={int(test_seconds.isna().sum())}"
        )

    normalized = raw.loc[:, REQUIRED].copy()
    normalized["Test_Time(s)"] = test_seconds
    normalized["Date_Time"] = parsed_timestamp
    return sheet_output_name, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-001-2', type=Path, default=SOURCES[0][0])
    parser.add_argument('--input-002-4', type=Path, default=SOURCES[1][0])
    args = parser.parse_args()
    sources = [(args.input_001_2, SOURCES[0][1]), (args.input_002_4, SOURCES[1][1])]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepared = [normalize_source(path, sheet_name) for path, sheet_name in sources]

    workbook = Workbook(write_only=True)
    for sheet_name, frame in prepared:
        ws = workbook.create_sheet(sheet_name)
        headers = frame.columns.tolist()
        append_header(ws, headers)
        date_index = headers.index("Date_Time")
        for values in frame.itertuples(index=False, name=None):
            row = []
            for index, value in enumerate(values):
                cell = WriteOnlyCell(ws, value=excel_value(value))
                cell.font = Font(name="Arial", size=9)
                if index == date_index and cell.value is not None:
                    cell.number_format = "yyyy-mm-dd hh:mm:ss"
                row.append(cell)
            ws.append(row)
        widths = [14] * len(headers)
        if "Date_Time" in headers:
            widths[headers.index("Date_Time")] = 21
        configure_sheet(ws, headers, widths)
    workbook.save(OUTPUT_FILE)
    from battery_config import CONFIG, CONFIG_SHA256
    manifest = {'configuration': CONFIG, 'configuration_sha256': CONFIG_SHA256,
                'sources': [{'path': str(path.resolve()),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'rows': len(frame), 'source_time_values': 'preserved in source workbook; parsed values in normalized output'}
                            for (path, _), (_, frame) in zip(sources, prepared)]}
    (OUTPUT_DIR / 'normalization_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"Created {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
