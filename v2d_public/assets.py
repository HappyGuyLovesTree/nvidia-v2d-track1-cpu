"""Explicit fixed-asset plans, bounded downloads, hash verification and safe ZIP extraction.

No download occurs on import, plan, or verify. Models are never loaded here.
"""
from pathlib import Path
from urllib.parse import urlsplit
import argparse, hashlib, json, os, shutil, stat, time, urllib.request, zipfile
from .common import ContractError, load, now, read_paths, ref, require, run_logged, save_new, sha, token

def verify_asset(asset,path):
    p=Path(path);size=p.stat().st_size if p.is_file() else None
    if asset.get('archive_kind')=='kit_zip' and p.is_dir():
        checks={name:(p/name).is_file() and sha(p/name)==value for name,value in asset['required_files'].items()}
        return {'key':asset['key'],'verified':all(checks.values()),'required_file_checks':checks,'identity':'extracted fixed kit members; archive identity checked at unpack'}
    value=sha(p) if p.is_file() else None
    return {'key':asset['key'],'exists':p.is_file(),'actual_bytes':size,'actual_sha256':value,
        'verified':p.is_file() and value==asset['sha256'] and (asset.get('bytes') is None or size==asset['bytes'])}

def download(asset,path,*,timeout=180,opener=urllib.request.urlopen):
    """One bounded request; partial failure is retained as evidence, never retried."""
    url=asset['url'];u=urlsplit(url);require(u.scheme=='https' and u.hostname and not u.username and not u.password,'asset URL must be HTTPS without embedded credentials')
    require(asset.get('bytes',0)>0,'download requires a fixed byte count')
    path=Path(path);require(not path.exists(),'download target exists; use verify or an explicit new workspace')
    path.parent.mkdir(parents=True,exist_ok=True);part=path.with_name(path.name+'.part_'+token())
    h=hashlib.sha256();total=0;start=time.monotonic()
    # A socket read timeout and a total monotonic limit bound this one request.
    with opener(urllib.request.Request(url,headers={'User-Agent':'v2d-public-reproduction/1'}),timeout=min(timeout,30)) as response,part.open('xb') as f:
        require(response.status==200,'asset request did not return HTTP 200')
        require(urlsplit(response.geturl()).scheme=='https','asset redirected outside HTTPS')
        encoding=response.headers.get('Content-Encoding','identity')
        require(encoding=='identity','asset response changed transfer representation')
        while True:
            require(time.monotonic()-start<timeout,'asset total transfer limit exceeded')
            block=response.read(min(1024*1024,asset['bytes']-total+1))
            if not block:break
            total+=len(block);require(total<=asset['bytes'],'asset response exceeds declared size');h.update(block);f.write(block)
        f.flush();os.fsync(f.fileno())
    require(total==asset['bytes'] and h.hexdigest()==asset['sha256'],'asset downloaded bytes/hash mismatch')
    # Hard-link create is exclusive, including a target created during download.
    os.link(part,path);part.unlink()
    return {'key':asset['key'],'source_url':url,'status':'verified','output':ref(path),'elapsed_s':time.monotonic()-start,'attempts':1,'model_loaded':False}

def extract_zip(archive,destination,*,archive_sha256,member=None,member_sha256=None,member_bytes=None,max_bytes=256*1024*1024):
    archive=Path(archive);destination=Path(destination)
    require(sha(archive)==archive_sha256,'ZIP archive hash mismatch')
    require(not destination.exists(),'ZIP output exists')
    staging=destination.with_name(destination.name+'.extract_'+token())
    staging.mkdir(parents=True,exist_ok=False)
    outputs=[];total=0
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist()
        require(len({i.filename for i in entries})==len(entries),'duplicate ZIP entry')
        if member is not None:
            require(member in {i.filename for i in entries},'ZIP member missing')
            entries=[z.getinfo(member)]
        # Validate the full selected plan before writing any member.
        for info in entries:
            rel=Path(info.filename);mode=info.external_attr>>16
            require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in info.filename,'unsafe ZIP path')
            require(not stat.S_ISLNK(mode),'ZIP symlink not allowed')
            total+=info.file_size
        require(total<=max_bytes,'ZIP declared extraction exceeds byte cap')
        written=0
        for info in entries:
            if info.is_dir():continue
            target=(staging/('asset' if member is not None else info.filename)).resolve()
            require(target.is_relative_to(staging.resolve()),'ZIP path escapes extraction directory')
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source,target.open('xb') as f:
                count=0
                while True:
                    block=source.read(1024*1024)
                    if not block:break
                    count+=len(block);written+=len(block)
                    require(count<=info.file_size and written<=max_bytes,'ZIP actual extraction exceeds byte cap');f.write(block)
            require(count==info.file_size,'ZIP member size mismatch')
            if member is not None:
                require(member_bytes==count and sha(target)==member_sha256,'ZIP target member content mismatch')
            outputs.append(target.relative_to(staging).as_posix())
    if member is not None:
        destination.parent.mkdir(parents=True,exist_ok=True)
        os.link(staging/'asset',destination);(staging/'asset').unlink();staging.rmdir()
    else:
        # mkdir+link refuses an existing destination even if empty.
        destination.mkdir(parents=True,exist_ok=False)
        for rel in outputs:
            target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);os.link(staging/rel,target)
        shutil.rmtree(staging)
    return {'archive':ref(archive),'destination':str(destination.resolve()),'member':member,'extracted_bytes':written,'files':outputs}

def download_curl(asset,path,log_directory,*,timeout=180):
    """macOS system TLS path. No curlrc, retries, credentials, or HTTP downgrade."""
    u=urlsplit(asset['url']);require(u.scheme=='https' and u.hostname and not u.username and not u.password,'asset URL must be HTTPS without embedded credentials')
    require(asset.get('bytes',0)>0 and timeout>0,'fixed positive bytes and timeout required')
    path=Path(path);require(not path.exists(),'download target exists; verify it instead')
    path.parent.mkdir(parents=True,exist_ok=True);part=path.with_name(path.name+'.part_'+token())
    curl=shutil.which('curl');require(curl,'curl executable missing')
    cmd=[curl,'-q','--fail','--silent','--show-error','--location','--proto','=https','--proto-redir','=https','--retry','0','--connect-timeout','15','--max-time',str(timeout),'--max-filesize',str(asset['bytes']),'--output',str(part),'--write-out','%{http_code}',asset['url']]
    run=run_logged(cmd,log_directory,timeout=timeout+5,max_output_bytes=1024*1024)
    require(run['status']=='completed' and Path(run['stdout']['path']).read_text()=='200','fixed asset HTTPS transfer failed')
    require(part.stat().st_size==asset['bytes'] and sha(part)==asset['sha256'],'downloaded asset bytes/hash mismatch')
    os.link(part,path);part.unlink()
    return {'key':asset['key'],'source_url':asset['url'],'status':'verified','output':ref(path),'curl':ref(curl),'execution':run,'attempts':1,'model_loaded':False}

def asset_plan(manifest,roster=None):
    rows=list(manifest['assets'])
    if roster:
        for r in roster['episodes']:
            rows.append({'key':f'video_{r["episode"]:06d}','kind':'external_video','url':r['download_url'],'bytes':r['declared_size_bytes'],'sha256':r['expected_sha256'],'logical_path':'assets/videos/'+r['repository_path']})
    require(len({r['key'] for r in rows})==len(rows),'duplicate asset key')
    return rows

def main(argv=None):
    p=argparse.ArgumentParser(description='Fixed assets: plan/verify are offline; fetch/unpack are explicit separate actions.')
    p.add_argument('mode',choices=['plan','verify','fetch','unpack']);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--roster',type=Path);p.add_argument('--paths',type=Path,required=True);p.add_argument('--workspace');p.add_argument('--key',action='append');p.add_argument('--timeout',type=float,default=180)
    a=p.parse_args(argv);paths,_=read_paths(a.paths,a.workspace);rows=asset_plan(load(a.manifest),load(a.roster) if a.roster else None)
    selected=set(a.key) if a.key else {r['key'] for r in rows}
    require(selected<={r['key'] for r in rows},'unknown asset key')
    require(a.mode not in ('fetch','unpack') or a.key,'fetch/unpack require explicit --key; no implicit full asset download')
    results=[]
    for row in rows:
        if row['key'] not in selected:continue
        target=paths.asset(row['key'],row['logical_path'])
        if a.mode=='plan':result={**row,'target':str(target)}
        elif a.mode=='verify':result=verify_asset(row,target)
        elif a.mode=='fetch':
            require(row.get('kind')!='kit_member','fetch and unpack the fixed kit to obtain this member')
            fetch_row=dict(row)
            if row.get('archive_member'):
                fetch_row.update(bytes=row['archive_bytes'],sha256=row['archive_sha256']);target=paths.output('downloads/'+row['key']+'.zip')
            elif row.get('archive_kind')=='kit_zip':target=paths.output('downloads/kit.zip')
            result=download_curl(fetch_row,target,paths.output('preparation/asset_transfer_'+token()),timeout=a.timeout)
        else:
            if row.get('archive_member'):
                result=extract_zip(paths.output('downloads/'+row['key']+'.zip'),target,archive_sha256=row['archive_sha256'],member=row['archive_member'],member_sha256=row['sha256'],member_bytes=row['bytes'],max_bytes=row['bytes'])
            else:
                require(row.get('archive_kind')=='kit_zip','asset is not an unpackable archive')
                # The official ZIP already contains the v2d_submission_kit top directory.
                result=extract_zip(paths.output('downloads/kit.zip'),paths.output('external'),archive_sha256=row['sha256'])
        results.append(result)
    record={'schema':'r12_public_asset_operation_v1','created_at':now(),'mode':a.mode,'results':results,'model_loaded':False}
    if a.mode!='plan':record['receipt']=save_new(paths.output('preparation/assets_'+token()+'.json'),record)
    print(json.dumps(record,ensure_ascii=False,indent=2))
    return 0 if a.mode!='verify' or all(r['verified'] for r in results) else 1
