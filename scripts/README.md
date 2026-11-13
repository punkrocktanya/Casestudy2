# Scripts: how to use

Run commands from the project root (the folder containing requirements.txt and outputs). Install dependencies first:

    python -m pip install -r requirements.txt

Core analysis, using the included processed data:

    python scripts/fit_capacity_fade.py
    python scripts/compare_capacity_models.py
    python scripts/differential_voltage.py
    python scripts/detect_battery_anomalies.py
    python scripts/individual_curve_rul.py
    python scripts/estimate_rul_xgboost_shared.py
    python scripts/estimate_rul_lstm_minimal.py
    python scripts/build_common_checkpoint_comparison.py
    python scripts/scale_module_pack.py
    python scripts/build_task6_visualizations.py
    python scripts/build_short_report.py

The current five-page report is output/pdf/Technical_Report_Module_Outputs.pdf. The report generator reads the saved outputs; reproduce dependent results before rebuilding. The raw Excel workbooks are absent, so raw preprocessing requires authorized source files and configuration. Review the numbered scripts and the project README before running preprocessing. Temperature is missing. The RUL endpoint is relative to cycle-2 capacity.

Tests: `python -m unittest discover -s scripts -p "test_*.py"`. Integrity check: `python scripts/validate_submission_integrity.py`. The script copies use the project root for data and outputs. The source Excel files and several historical supplementary reports are not included. Model and visualization manifests preserve historical source hashes and do not attest the relocated code. Run `python scripts/package_submission.py` to create an allowlisted ZIP after checking the desired outputs.

## Markdown result records for scripts

Run any Python script through the common runner from the project root:

```bash
python scripts/run_with_report.py fit_capacity_fade.py
python scripts/run_with_report.py differential_voltage.py
python scripts/run_with_report.py estimate_rul_xgboost_shared.py
python scripts/run_with_report.py estimate_rul_lstm_minimal.py
python scripts/run_with_report.py build_short_report.py
```

The runner forwards arguments after the script name. It saves a timestamped Markdown record in `outputs/execution_reports/`, with status, console output, links to changed outputs and previews of generated result tables where available. Tests and library utilities may not produce scientific findings. The record reports what ran; it does not invent interpretations. Direct calls such as `python scripts/fit_capacity_fade.py` retain their original behavior and do not create these records.

## PDF generation

After generating source CSVs and figures, run `python scripts/run_with_report.py build_short_report.py` to generate `output/pdf/Technical_Report_Module_Outputs.pdf` and a Markdown execution record. Other PDF builders are `build_battery_report.py`, `build_limitations_report.py`, `build_rul_comprehensive_report.py`, and `build_task6_visualizations.py`; run each with the same runner. Install `requirements.txt` first. Some historical PDFs were absent from the original ZIP and must be generated before use. Builders may overwrite same-named outputs.

GitHub: Tests runs on push/PR; start Full analysis and report manually under Actions. Generated records, plots and PDFs are downloadable workflow artifacts. The final report remains in the repository root as Technical_Report.pdf.
