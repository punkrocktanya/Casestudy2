"""Build an allowlisted repository ZIP without raw Excel, credentials or dependencies."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'deliverables'
TOP={'README.md','requirements.txt','battery_thresholds.json','.gitignore','TASK3_MODEL_GUIDE.md','battery_specification.md'}
DIRS=['outputs/capacity_fade','outputs/capacity_comparison_local','outputs/differential_voltage/revised',
      'outputs/anomaly_detection','outputs/rul_individual_curves','outputs/rul_xgboost_shared','outputs/rul_lstm_minimal',
      'outputs/reporting_visualization','outputs/rul_all_models_validation','outputs/pack_scaling','outputs/rul_common_checkpoint','outputs/validation','output/pdf','notebooks','.github']
SUFFIXES={'.py','.md','.txt','.json','.csv','.png','.pdf','.ipynb','.yml','.yaml','.pt','.ubj'}


def files():
    selected=set((ROOT/'scripts').glob('*.py'))|{ROOT/'scripts/README.md'}|{ROOT/n for n in TOP}
    for directory in DIRS:
        selected.update(p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in SUFFIXES)
    selected.update((ROOT/'outputs/silver').glob('*_silver.parquet'))
    selected.update(p for p in (ROOT/'outputs/gold').glob('*') if p.is_file() and not p.name.startswith('interval_diagnostics_') and p.suffix in {'.csv','.json','.md','.parquet'})
    selected.update(p for p in (ROOT/'outputs/battery_cleaning').glob('*') if p.suffix in {'.md','.json'})
    selected.update(p for p in (ROOT/'outputs/silver').glob('*') if p.suffix in {'.md','.json'})
    updated = ROOT/'output/pdf/Task_6_Reporting_and_Visualization_Updated.pdf'
    if updated.exists():
        selected.discard(ROOT/'output/pdf/Task_6_Reporting_and_Visualization.pdf')
    return sorted(p for p in selected if p.is_file())


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    selected=files()
    records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in selected]
    manifest=dict(files=records,raw_excel_included=False,remote_publication=False,report_max_pages=5,
                  note='Review data-sharing permissions before publishing derived data. No license inferred.')
    (DEST/'submission_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    path=DEST/'Battery_Analysis_Submission.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in selected:archive.write(p,'battery-analysis/'+p.relative_to(ROOT).as_posix())
        archive.writestr('battery-analysis/SUBMISSION_MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert not any('/.git/' in s or '.local_deps/' in s or s.endswith('.xlsx') for s in archive.namelist())
    print(f'{path}: {len(selected)} files; {path.stat().st_size/1024/1024:.2f} MiB')


if __name__=='__main__':main()
