"""Run a project script and record its results and execution in Markdown."""
from pathlib import Path
import argparse, subprocess, sys, datetime, csv
ROOT=Path(__file__).resolve().parent.parent
SCRIPTS=ROOT/'scripts'
REPORTS=ROOT/'outputs'/'execution_reports'

def snapshot():
    result={}
    for folder in [ROOT/'outputs',ROOT/'output',ROOT/'deliverables']:
        if folder.exists():
            for p in folder.rglob('*'):
                if p.is_file() and REPORTS not in p.parents:
                    st=p.stat();result[p.relative_to(ROOT).as_posix()]=(st.st_mtime_ns,st.st_size)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script',help='Python filename in scripts/')
    parser.add_argument('script_args',nargs=argparse.REMAINDER)
    args=parser.parse_args();target=(SCRIPTS/args.script).resolve()
    if target.parent!=SCRIPTS.resolve() or target.suffix!='.py' or not target.is_file() or target==Path(__file__).resolve():
        parser.error('Choose an existing Python file in scripts/, excluding this runner')
    REPORTS.mkdir(parents=True,exist_ok=True);before=snapshot()
    try:
        p=subprocess.run([sys.executable,str(target),*args.script_args],cwd=ROOT,text=True,encoding='utf-8',errors='replace',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        status,output=p.returncode,p.stdout
    except OSError as e:status,output=1,str(e)
    after=snapshot();changed=[n for n in sorted(after) if before.get(n)!=after[n]]
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report=REPORTS/(target.stem+'_'+stamp+'.md')
    def esc(v):return str(v).replace('|','\\|').replace('\n',' ')
    lines=['# Result: '+target.name,'',f'- Status: {"SUCCESS" if status==0 else "FAILED"} (exit {status})',f'- UTC: {stamp}',f'- Script arguments: `{args.script_args}`',f'- Changed output files: {len(changed)}','','## Output files','']
    if changed:lines.extend('- ['+esc(n)+'](../../'+n.replace(' ','%20')+')' for n in changed[:100])
    else:lines.append('No changed files detected in outputs/, output/, or deliverables/. See console output; libraries may have no standalone result.')
    if len(changed)>100:lines.append(f'- Plus {len(changed)-100} additional files.')
    metrics=[n for n in changed if n.endswith('.csv') and any(k in n.lower() for k in ['metric','summary','prediction'])]
    if metrics:lines+=['','## Result previews','','Unmodified previews of generated CSVs; interpret with the method and data limitations.']
    for n in metrics[:3]:
        try:
            with (ROOT/n).open(encoding='utf-8-sig',newline='') as f:
                rows=[]
                for i,row in enumerate(csv.reader(f)):
                    if i>=6:break
                    rows.append(row[:8])
            if rows and rows[0]:
                w=len(rows[0]);lines+=['','**'+esc(n)+'**','','| '+' | '.join(map(esc,rows[0]))+' |','| '+' | '.join(['---']*w)+' |']
                for row in rows[1:]:lines.append('| '+' | '.join(esc(v) for v in (row+['']*w)[:w])+' |')
        except (OSError,UnicodeError,csv.Error):pass
    lines+=['','## Script output','','```text',output[:60000].replace('```','[code fence]'),'```']
    if len(output)>60000:lines+=['','Output truncated at 60,000 characters.']
    lines+=['','## Interpretation','','This record describes the execution and previews available results. A successful run does not establish scientific validity; inspect the outputs, assumptions and validation.']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if output:print(output,end='' if output.endswith('\n') else '\n')
    print('Markdown result:',report)
    return status
if __name__=='__main__':sys.exit(main())
