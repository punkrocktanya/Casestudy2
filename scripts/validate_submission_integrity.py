"""Verify packaged file hashes, required artifacts, and source syntax."""
from pathlib import Path
import ast, hashlib, json
ROOT=Path(__file__).resolve().parent.parent

def main():
    manifest=json.loads((ROOT/'SUBMISSION_MANIFEST.json').read_text(encoding='utf-8'))
    errors=[]
    for record in manifest['files']:
        p=ROOT/record['path']
        if not p.is_file(): errors.append('Missing: '+record['path']); continue
        if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']: errors.append('Changed: '+record['path'])
    for p in (ROOT/'scripts').glob('*.py'):
        try: ast.parse(p.read_text(encoding='utf-8-sig'))
        except (SyntaxError, UnicodeError) as e: errors.append(f'{p}: {e}')
    if errors: raise SystemExit('Integrity errors:\n'+'\n'.join(errors))
    print(f"Verified {len(manifest['files'])} files and Python syntax. Hashes do not validate scientific results.")
if __name__=='__main__': main()
