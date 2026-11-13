#!/usr/bin/env python3
"""Run source normalization, cleaning, Silver and validated Gold in order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(script_name: str):
    script = ROOT / "scripts" / script_name
    print(f"Running {script.name}...")
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def main():
    run("01_normalize_timestamps.py")
    run("02_clean_and_validate.py")
    run("02_silver_transform.py")
    run("03_build_cycle_features.py")
    print("Battery pipeline completed; see outputs/gold/corrective_action_report.md for release limits.")


if __name__ == "__main__":
    main()
