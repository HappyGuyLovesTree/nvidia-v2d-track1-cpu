"""Public Track1 sample, source identity and full-PTS preparation; no model imports.

PTS predicates follow frozen T011.05 fetcher.validate_pts (original project code,
SHA256 206c5699cc2fffa4fc82b451abf2b7032eb1cb28d4b49b8427d11ca987875773).
Added explicit workspace, new evidence identities and independently supplied
official bindings. Historical evidence is never rewritten or fabricated.
"""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import argparse, csv, json, re, shutil
from .common import ContractError, Paths, digest, load, now, read_paths, ref, require, run_logged, save_new, sha, token

IDENTITY=('source_repository','revision','repository_path','declared_size_bytes','expected_sha256')
SAMPLE_SHA='1db56c5ec7267cdb7a68d42928f85c092c6b8e9bfbc0dc98a2b020d7a48bc621'

def segments(required):
    require(required and required==sorted(set(required)) and all(type(f) is int and f>=0 for f in required),'required frame IDs must be sorted unique nonnegative integers')
    runs=[]
    for f in required:
        if not runs or f!=runs[-1][-1]+1:runs.append([])
        runs[-1].append(f)
    return runs

def sample_rows(path):
    path=Path(path)
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path,columns=['row_id']).column('row_id').to_pylist()
    require(path.suffix=='.csv','sample must be Parquet or CSV')
    with path.open(newline='') as f:
        reader=csv.DictReader(f);require('row_id' in (reader.fieldnames or []),'sample lacks row_id')
        return [r['row_id'] for r in reader]

def required_from_keys(rows):
    require(len(rows)==len(set(rows)) and rows,'empty or duplicate sample row_id')
    result=defaultdict(set)
    for row in rows:
        require(isinstance(row,str) and re.fullmatch(r't1_\d{6}_\d{6}_[0-7]_\d{6}',row),'invalid Track1 row_id')
        _,e,f,a,p=row.split('_');e,f,a,p=map(int,(e,f,a,p))
        result[e]
        if f!=999999:
            require(a in (0,3,4),'per-episode array has a real-frame row')
            result[e].add(f)
        else:require(a in (1,2,5,6,7),'per-frame array has an episode-marker row')
    return {e:sorted(v) for e,v in result.items()}

def validate_roster(roster, rows=None, scope='official'):
    require(scope in ('official','fixture'),'invalid scope')
    episodes=roster.get('episodes',[]);ids=[r['episode'] for r in episodes]
    require(ids==sorted(set(ids)) and ids,'invalid roster episode order/set')
    for r in episodes:
        require(all(k in r for k in (*IDENTITY,'declared_video','declared_fps','declared_frame_count','required_original_frames','required_frame_stretches')),'roster lacks source or stream fields')
        require(re.fullmatch('[0-9a-f]{64}',r['expected_sha256']) is not None,'invalid video content hash')
        runs=segments(r['required_original_frames'])
        require(max(r['required_original_frames'])<r['declared_frame_count'],'required frame exceeds declared video')
        require(r['required_frame_stretches']==[{'first':s[0],'last':s[-1],'count':len(s)} for s in runs],'roster run declarations differ')
    if scope=='official':
        require(ids==list(range(30)) and sum(len(r['required_original_frames']) for r in episodes)==9877,'official roster coverage mismatch')
        require(sum(len(segments(r['required_original_frames'])) for r in episodes)==65,'official source count mismatch')
        require(all(r['source_repository']=='nvidia/video_to_data_challenge' and r['revision']=='5f68335f3acc802033d1e80728c1633197521de8' and r['repository_path']==f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{r["episode"]:06d}.mp4' for r in episodes),'official video source identity mismatch')
    if rows is not None:
        required=required_from_keys(rows)
        require(required=={r['episode']:r['required_original_frames'] for r in episodes},'sample and roster original-frame sets differ')
        require(len(rows)==sum(50*len(v)+8231 for v in required.values()),'sample row budget mismatch')
        if scope=='official':require(len(rows)==740780,'official sample row count mismatch')
    return episodes

def verify_file(row,path):
    p=Path(path);identity={k:row[k] for k in IDENTITY}
    exists=p.is_file();size=p.stat().st_size if exists else None
    actual=sha(p) if exists else None
    return {'path':str(p.resolve()),'identity':identity,'exists':exists,'actual_size_bytes':size,'actual_sha256':actual,
            'file_verified':exists and size==row['declared_size_bytes'] and actual==row['expected_sha256']}

def validate_pts(row,data,exit_code,stderr):
    frames=data.get('frames',[]);streams=data.get('streams',[]);pts=[f.get('pts') for f in frames]
    present=bool(frames) and all(type(p) is int for p in pts)
    mono=present and all(b>a for a,b in zip(pts,pts[1:]))
    stream=streams[0] if len(streams)==1 else {};decl=row['declared_video']
    stream_ok=all(stream.get(k)==decl['video.'+dk] for k,dk in [('width','width'),('height','height'),('codec_name','codec'),('pix_fmt','pix_fmt')])
    intervals=False
    try:
        tb=Fraction(stream['time_base']);fps=Fraction(str(row['declared_fps']))
        intervals=bool(mono and tb>0 and all(Fraction(b-a)*tb==1/fps for a,b in zip(pts,pts[1:])))
    except (ValueError,KeyError,ZeroDivisionError):pass
    required=row['required_original_frames'];missing=[f for f in required if not 0<=f<len(frames)]
    count_ok=len(frames)==row['declared_frame_count']
    passed=bool(exit_code==0 and not stderr.strip() and mono and count_ok and not missing and stream_ok and intervals)
    return {'whole_pts_verified':passed,'stream':stream,'decoded_frame_count':len(frames),'declared_frame_count':row['declared_frame_count'],
        'pts_present':present,'pts_strictly_monotonic':mono,'frame_count_matches_declaration':count_ok,'stream_matches_declaration':stream_ok,
        'all_intervals_match_declared_fps':intervals,'required_count':len(required),'required_covered':len(required)-len(missing),'missing_required_frames':missing,
        'frame_map':[{'original_frame':i,'pts':p,'best_effort_timestamp':f.get('best_effort_timestamp'),'pkt_duration':f.get('pkt_duration')} for i,(p,f) in enumerate(zip(pts,frames))]}

def official_binding(row,bindings):
    """Consume normalized external evidence. Local file hashing cannot set this true."""
    if bindings is None:return {'result':'pending','official_independent_binding':False,'reason':'no official binding evidence supplied'}
    entries=[x for x in bindings.get('entries',[]) if x.get('episode')==row['episode']]
    require(len(entries)==1,'official binding missing or duplicate episode')
    b=entries[0]
    require(all(b.get(k)==row[k] for k in ('source_repository','revision','repository_path','declared_size_bytes')) and b.get('lfs_content_sha256')==row['expected_sha256'],'official source binding conflicts with fixed declaration')
    require(b.get('result')=='passed' and b.get('evidence_refs'),'official source binding not passed or has no response evidence')
    for r in b['evidence_refs']:
        require(Path(r['path']).is_file() and sha(r['path'])==r['sha256'],'official binding response evidence missing or changed')
    return {'result':'passed','official_independent_binding':True,'evidence_refs':b['evidence_refs'],'binding_digest':digest(b)}

def prepare_one(row,path,paths,*,ffprobe,timeout=120,bindings=None,scope='official',roster_ref=None):
    require(scope in ('official','fixture'),'invalid scope')
    root=paths.output('preparation/'+f'episode_{row["episode"]:06d}_'+token());root.mkdir(parents=True,exist_ok=False)
    record={'schema':'r12_public_whole_pts_v1','scope':scope,'episode':row['episode'],'created_at':now(),'status':'failed','whole_pts_verified':False,
        'file_verification':None,'roster_ref':roster_ref,'checker_source':ref(Path(__file__)),'official_binding':None,'numeric_inference_performed':False}
    try:
        record['file_verification']=verify_file(row,path)
        require(record['file_verification']['file_verified'],'video file size/content hash mismatch')
        binding=official_binding(row,bindings);record['official_binding']=binding
        executable=Path(ffprobe).expanduser().resolve();require(executable.is_file(),'ffprobe executable missing')
        cmd=[str(executable),'-v','error','-select_streams','v:0','-show_frames','-show_streams','-show_entries',
             'stream=index,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,time_base,nb_frames:frame=pts,best_effort_timestamp,pkt_duration','-of','json',str(Path(path).resolve())]
        run=run_logged(cmd,root/'probe',timeout=timeout,max_output_bytes=8*1024*1024);record['probe']=run;record['ffprobe']=ref(executable)
        require(run['status']=='completed','ffprobe process did not complete successfully')
        stderr=Path(run['stderr']['path']).read_text();raw=load(run['stdout']['path'])
        record.update(validate_pts(row,raw,run['returncode'],stderr))
        require(sha(path)==record['file_verification']['actual_sha256'],'video changed during PTS probe')
        require(record['whole_pts_verified'],'whole original-frame PTS validation failed')
        runs=segments(row['required_original_frames'])
        episode={'schema':'r12_public_episode_input_v1','scope':scope,'episode':row['episode'],'source_identity':{k:row[k] for k in IDENTITY},
             'video':ref(path),'required':row['required_original_frames'],'required_digest':digest(row['required_original_frames']),
             'sources':runs,'source_map':{str(f):i for i,s in enumerate(runs) for f in s},'stream':record['stream'],'decoded_frame_count':record['decoded_frame_count'],
             'official_independent_binding':binding['official_independent_binding'],'local_file_verified':True,'whole_pts_verified':True,
             'recipe_commit':None,'actual_prediction_producer_commit':None,'fresh_session_or_budget_created':False}
        # Receipt is final before its immutable hash is bound into the episode input.
        record['status']='passed';record['finished_at']=now();rr=save_new(root/'whole_pts.json',record)
        episode['whole_pts']=rr;er=save_new(root/'episode_input.json',episode)
        return {'episode':row['episode'],'status':'passed','scope':scope,'whole_pts':rr,'episode_input':er,'official_independent_binding':binding['official_independent_binding'],'submission_ready':False}
    except Exception as exc:
        record.update(status='failed',whole_pts_verified=False,error={'type':type(exc).__name__,'message':str(exc)},finished_at=now())
        rr=save_new(root/('preparation_failure.json' if (root/'whole_pts.json').exists() else 'whole_pts.json'),record)
        return {'episode':row['episode'],'status':'failed','scope':scope,'whole_pts':rr,'error':record['error'],'submission_ready':False}

def main(argv=None):
    p=argparse.ArgumentParser(description='Prepare public Track1 inputs and full original-frame PTS; no prediction model calls.')
    p.add_argument('mode',choices=['plan','verify','pts']);p.add_argument('--roster',type=Path,required=True);p.add_argument('--paths',type=Path,required=True);p.add_argument('--workspace');p.add_argument('--sample',type=Path);p.add_argument('--episode',type=int,action='append');p.add_argument('--bindings',type=Path);p.add_argument('--ffprobe',default=shutil.which('ffprobe'));p.add_argument('--timeout',type=float,default=120);p.add_argument('--scope',choices=['official','fixture'],default='official')
    a=p.parse_args(argv);paths,_=read_paths(a.paths,a.workspace);roster=load(a.roster)
    rows=sample_rows(a.sample) if a.sample else None
    if a.scope=='official' and a.mode=='pts':require(a.sample and sha(a.sample)==SAMPLE_SHA,'official PTS requires the fixed full sample')
    allrows=validate_roster(roster,rows,a.scope);selected=set(a.episode) if a.episode else {r['episode'] for r in allrows}
    require(selected<={r['episode'] for r in allrows},'unknown selected episode')
    results=[]
    bindings=load(a.bindings) if a.bindings else None
    for row in allrows:
        if row['episode'] not in selected:continue
        video=paths.asset(f'video_{row["episode"]:06d}','assets/videos/'+row['repository_path'])
        if a.mode=='plan':result={'episode':row['episode'],'url':row['download_url'],'target':str(video),'sha256':row['expected_sha256'],'bytes':row['declared_size_bytes']}
        elif a.mode=='verify':result={'episode':row['episode'],**verify_file(row,video)}
        else:result=prepare_one(row,video,paths,ffprobe=a.ffprobe,timeout=a.timeout,bindings=bindings,scope=a.scope,roster_ref=ref(a.roster))
        results.append(result)
    record={'schema':'r12_public_input_preparation_v1','created_at':now(),'mode':a.mode,'scope':a.scope,'roster':ref(a.roster),'sample':ref(a.sample) if a.sample else None,'results':results,'model_calls':0,'submission_ready':False}
    if a.mode!='plan':record['receipt']=save_new(paths.output('preparation/input_index_'+token()+'.json'),record)
    print(json.dumps(record,ensure_ascii=False,indent=2))
    return 0 if a.mode=='plan' or all(r.get('file_verified',r.get('status')=='passed') for r in results) else 1
