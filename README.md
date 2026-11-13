# Battery degradation analysis

Two-cell exploratory analysis of cycle data, capacity fade, anomaly screening, relative RUL and module/pack scaling. The five-page report is [Technical_Report.pdf](Technical_Report.pdf) in the repository root; a regenerated copy is written to `output/pdf/Technical_Report_Module_Outputs.pdf`.

All Python scripts are in `scripts/`; see [scripts/README.md](scripts/README.md) for installation and run order. Processed Silver and Gold data and saved results are in `outputs/`; source Excel workbooks are not supplied. Install with `python -m pip install -r requirements.txt`.

The exploratory EOL threshold is 80% of each cell's cycle-2 delivered capacity, not verified nominal SOH. Temperature is missing; only two cells were tested at different currents. Pack conclusions are illustrative or proposed, not validated pack performance.

The root `SUBMISSION_MANIFEST.json` records hashes of packaged files. Run `python scripts/validate_submission_integrity.py` before modifying outputs. The individual model and visualization manifests record the earlier experiment; their script hashes refer to scripts before relocation and are retained as provenance, not as checks of the relocated sources.

See `scripts/README.md` for Markdown execution records and PDF generation instructions.

## GitHub Actions

The **Tests** workflow runs on pushes and pull requests. The **Full analysis and report** workflow is manually started from Actions and uploads Markdown execution records, figures and generated PDFs as an artifact. It uses packaged processed data; it cannot regenerate raw preprocessing without the original workbooks. It does not commit regenerated outputs or replace the root report. Workflows have not been run on GitHub yet. Saved outputs remain in this copy; review large artifacts before publishing.
