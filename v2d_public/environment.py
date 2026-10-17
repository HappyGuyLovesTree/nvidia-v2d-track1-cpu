"""Check distribution metadata and active dependency lock without importing models."""
from pathlib import Path
import argparse, base64, importlib.metadata as md, json, platform, sys
from .common import load, ref, require, sha

def main(argv=None):
    p=argparse.ArgumentParser(description='Check the fixed macOS arm64/Python3.12.7 distribution lock; no model imports.')
    p.add_argument('--lock',type=Path,required=True);p.add_argument('--extra-site',action='append',default=[])
    p.add_argument('--strict-installed-record',action='store_true',help='also require the same installed RECORD bytes; clean installs may regenerate this file')
    p.add_argument('--verify-files',action='store_true',help='hash all files covered by each installed RECORD; still no model imports')
    a=p.parse_args(argv)
    for path in reversed(a.extra_site):sys.path.insert(0,str(Path(path).resolve()))
    lock=load(a.lock);require(lock['complete_wheel_lock'],'wheel lock is unresolved')
    issues=[];checks=[]
    if sys.version_info[:3]!=(3,12,7):issues.append('Python must be3.12.7 for this recorded numerical environment')
    if platform.system()!='Darwin' or platform.machine()!='arm64':issues.append('this lock is only for macOS arm64')
    for row in lock['distributions']:
        try:
            d=md.distribution(row['name']);base=Path(d._path)
            same_record=sha(base/'RECORD')==row['record_sha256']
            ok=d.version==row['version'] and sha(base/'METADATA')==row['metadata_sha256'] and (same_record or not a.strict_installed_record)
            mismatches=[]
            if a.verify_files:
                for f in d.files or []:
                    if not f.hash:continue
                    path=d.locate_file(f)
                    actual=base64.urlsafe_b64encode(bytes.fromhex(sha(path))).decode().rstrip('=') if path.is_file() else None
                    if f.hash.mode!='sha256' or actual!=f.hash.value:mismatches.append(str(f))
                ok=ok and not mismatches
        except md.PackageNotFoundError:ok=False;same_record=False;mismatches=[]
        checks.append({'distribution':row['name'],'version':row['version'],'recorded_distribution_identity':ok,'same_original_installed_RECORD':same_record,'file_hashes_checked':a.verify_files,'file_mismatches':mismatches})
    print(json.dumps({'lock':ref(a.lock),'issues':issues,'checks':checks,'passed':not issues and all(c['recorded_distribution_identity'] for c in checks),
        'claim':'fixed versions/metadata, optional original RECORD and file checks; not a fresh installation test, model load or numerical replay'},ensure_ascii=False,indent=2))
    return 0 if not issues and all(c['recorded_distribution_identity'] for c in checks) else 1
