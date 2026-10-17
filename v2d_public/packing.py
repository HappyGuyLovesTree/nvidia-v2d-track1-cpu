"""Validate a reconstruction view and run the unchanged official packer as a child."""
from pathlib import Path
import argparse, json, os, re, subprocess, sys
from .common import load, now, read_paths, ref, require, run_logged, save_new, sha, token, digest
from .inputs import SAMPLE_SHA, required_from_keys, sample_rows

def verify_kit(kit,manifest):
    kit=Path(kit).resolve();expected=manifest['files']
    for name,value in expected.items():
        p=(kit/name).resolve();require(p.is_relative_to(kit),'kit manifest path escapes root')
        require(p.is_file() and sha(p)==value,'fixed official kit file differs: '+name)
    actual={p.relative_to(kit).as_posix():sha(p) for p in kit.rglob('*.py')}
    require(actual=={k:v for k,v in expected.items() if k.endswith('.py')},'kit Python file set differs from fixed snapshot')
    return {'kit':str(kit),'manifest_digest':digest(expected),'file_count':len(expected)}

def check_view(directory,rows,*,masks=None,provenance=None,scope='official',build_commit=None):
    import numpy as np
    directory=Path(directory).resolve();frames=required_from_keys(rows);files=[]
    if scope=='official':
        require(sorted(frames)==list(range(30)) and sum(map(len,frames.values()))==9877 and len(rows)==740780,'official complete sample coverage required')
        require(provenance and provenance.get('schema')=='r12_public_provenance_v1' and provenance.get('build_commit')==build_commit,'actual finalized provenance/build commit required')
        entries=provenance.get('episodes',[]);require([e['episode'] for e in entries]==list(range(30)),'provenance must contain exact ordered30 episodes')
        require(masks is not None,'official pack requires completion masks')
    else:entries=[]
    ep_refs={}
    for episode,ids in frames.items():
        npz=directory/f'episode_{episode:06d}.npz';obj=directory/f'episode_{episode:06d}_object.obj'
        require(npz.is_file() and obj.is_file(),'episode NPZ or same-name OBJ missing')
        require(not any(obj.with_suffix(s).exists() for s in ('.glb','.ply','.stl','.off')),'ambiguous supported mesh files')
        with np.load(npz,allow_pickle=False) as source:data={k:source[k] for k in source.files}
        needed={'pose','scales','shape','object_rotation','object_translation','object_scale'}
        require(set(data)==needed,'episode must contain the six exact arrays')
        T=len(data['pose']);require(T>max(ids),'arrays stop before a scored original frame')
        for key,dims in [('pose',(T,136)),('object_rotation',(T,3,3)),('object_translation',(T,3)),('scales',(68,)),('shape',(45,)),('object_scale',())]:
            require(data[key].shape==dims,'array dimensions: '+key)
            selected=data[key][ids] if key in ('pose','object_rotation','object_translation') else data[key]
            require(np.isfinite(selected).all(),'nonfinite required numerical values: '+key)
        require(data['object_scale']>0,'object scale must be positive scalar')
        R=data['object_rotation'][ids]
        require(np.abs(R.transpose(0,2,1)@R-np.eye(3)).max()<=1e-6 and np.abs(np.linalg.det(R)-1).max()<=1e-6,'invalid object rotations')
        refs={'npz':ref(npz),'obj':ref(obj)}
        if masks is not None:
            maskfile=Path(masks)/f'episode_{episode:06d}.npz'
            with np.load(maskfile,allow_pickle=False) as source:
                require(set(source.files)=={'mask','original_frames'},'mask file fields differ')
                mask=source['mask'];original=source['original_frames']
            require(mask.dtype==np.bool_ and mask.shape==(T,) and np.flatnonzero(mask).tolist()==ids and original.tolist()==ids,'completion mask/original frames mismatch')
            for key in ('pose','object_rotation','object_translation'):require(np.isnan(data[key][~mask]).all(),'uncomputed dense positions must be NaN')
            refs['mask']=ref(maskfile)
        if scope=='official':
            entry=entries[episode]
            require(re.fullmatch('[0-9a-f]{40}',entry.get('prediction_producer_commit','') or '') is not None and re.fullmatch('[0-9a-f]{40}',entry.get('producer_tree','') or '') is not None,'real per-episode producer/tree required')
            require(entry.get('pack_producer_commit')==build_commit and type(entry.get('imported')) is bool,'actual pack producer/import status required')
            require(entry.get('required_digest')==digest(ids),'provenance required-frame digest differs')
            require(re.fullmatch('[0-9a-f]{64}',entry.get('input_hash','') or '') is not None,'provenance input content hash missing')
            for key,r in refs.items():require(entry.get(key,{}).get('sha256')==r['sha256'],'provenance artifact hash differs: '+key)
        files.extend(refs.values());ep_refs[str(episode)]=refs
    return {'episode_count':len(frames),'required_frame_count':sum(map(len,frames.values())),'files':files,'per_episode':ep_refs}

def validate_candidate(path,rows,commit):
    import numpy as np
    import pandas as pd
    table=pd.read_parquet(path)
    require(table.columns.tolist()==['row_id','x','y','z','code_commit_url'],'candidate columns differ')
    require(table.row_id.tolist()==rows and table.row_id.is_unique,'candidate keys/order differ')
    require(np.isfinite(table[['x','y','z']].to_numpy()).all(),'candidate contains nonfinite values')
    require(table.code_commit_url.unique().tolist()==[commit],'candidate URL not uniform/exact')
    return {'rows':len(table),'columns':table.columns.tolist(),'exact_key_order':True,'finite':True,'uniform_url':commit,'output':ref(path)}

def build_identity(source,commit):
    def git(*args):return subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
    require(git('rev-parse','HEAD')==commit,'public build URL does not match actual checkout HEAD')
    require(git('status','--porcelain')=='','public build checkout must be clean')
    return {'commit':commit,'tree':git('rev-parse','HEAD^{tree}'),'dirty':False}

def main(argv=None):
    p=argparse.ArgumentParser(description='Run the unchanged pinned Track1 stock writer; never upload.')
    p.add_argument('mode',choices=['plan','run']);p.add_argument('--paths',type=Path,required=True);p.add_argument('--workspace');p.add_argument('--episodes',type=Path,required=True);p.add_argument('--sample',type=Path);p.add_argument('--masks',type=Path);p.add_argument('--provenance',type=Path);p.add_argument('--commit',required=True);p.add_argument('--build-source',type=Path);p.add_argument('--kit-manifest',type=Path,required=True);p.add_argument('--scope',choices=['official','fixture'],default='official');p.add_argument('--timeout',type=float,default=360)
    a=p.parse_args(argv);require(re.fullmatch(r'https://github\.com/[\w.-]+/[\w.-]+/commit/[0-9a-f]{40}',a.commit),'full40 actual GitHub commit URL required')
    commit=a.commit.rsplit('/',1)[1];paths,config=read_paths(a.paths,a.workspace)
    kit=paths.asset('kit','external/v2d_submission_kit');km=load(a.kit_manifest);verified_kit=verify_kit(kit,km)
    sample=a.sample or kit/'data/track_1_sample_submission.parquet'
    if a.scope=='official':require(sha(sample)==SAMPLE_SHA,'fixed full official sample required')
    rows=sample_rows(sample);provenance=load(a.provenance) if a.provenance else None
    identity=None
    if a.scope=='official':require(a.build_source is not None,'actual build source checkout required');identity=build_identity(a.build_source,commit)
    view=check_view(a.episodes,rows,masks=a.masks,provenance=provenance,scope=a.scope,build_commit=commit)
    root=paths.output('packs/'+token());out=root/'candidate.parquet'
    cmd=[sys.executable,'-B',str(kit/'tools/pack_reconstruction.py'),'track1','--episodes',str(a.episodes.resolve()),'--sample',str(sample.resolve()),'--commit',a.commit,'--out',str(out)]
    record={'schema':'r12_stock_pack_execution_v1','scope':a.scope,'created_at':now(),'status':'planned','python':ref(sys.executable),'cwd':str(kit),'kit':verified_kit,'kit_manifest':ref(a.kit_manifest),'packer':ref(kit/'tools/pack_reconstruction.py'),'sample':ref(sample),'input_view':view,'provenance':ref(a.provenance) if a.provenance else None,'build_code':identity,'pack_producer_commit':commit if identity else None,'uniform_url':a.commit,'argv':cmd,'publication_access':'not_checked_by_packer','submission_ready':False}
    if a.mode=='plan':print(json.dumps(record,ensure_ascii=False,indent=2));return 0
    root.mkdir(parents=True,exist_ok=False);save_new(root/'started.json',record)
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    extra=[str(Path(x).expanduser().resolve()) for x in config.get('python_extra_sites',[])]
    env['PYTHONPATH']=os.pathsep.join(extra)
    try:
        run=run_logged(cmd,root/'stock_process',timeout=a.timeout,max_output_bytes=32*1024*1024,env=env,cwd=kit);record['process']=run
        require(run['status']=='completed','stock writer failed; see execution receipt')
        require(all(sha(r['path'])==r['sha256'] for r in view['files']),'input artifact changed during stock packing')
        require(sha(sample)==record['sample']['sha256'],'sample changed during stock packing')
        if a.provenance:require(sha(a.provenance)==record['provenance']['sha256'],'provenance changed during stock packing')
        if identity:require(build_identity(a.build_source,commit)==identity,'build code changed during stock packing')
        verify_kit(kit,km)
        record['validation']=validate_candidate(out,rows,a.commit);record['status']='completed';record['official_writer_executed']=True
    except Exception as exc:record.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
    record['finished_at']=now();record['receipt']=save_new(root/'receipt.json',record)
    print(json.dumps(record,ensure_ascii=False,indent=2));return 0 if record['status']=='completed' else 1
