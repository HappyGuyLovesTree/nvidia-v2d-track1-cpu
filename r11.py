#!/usr/bin/env python3
"""Frozen R011 prepare / serial episodes / resume / aggregate / status CLI."""
from pathlib import Path
import argparse,datetime,json,os,signal,subprocess,sys,time,traceback
from common import load,sha
from tracking import digest
from r11_inputs import *
from r11_state import *

DEFAULT=P/'config_r11_20261017a1.json'
def next_dated(directory,stem,date):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);base=directory/f'{stem}_{date}.json'
    existing=list(directory.glob(f'{stem}_{date}*.json'))
    if not existing:return base
    order=[]
    for p in existing:
        tail=p.stem[len(stem)+1+len(date):]
        if tail=='':order.append(-1)
        elif len(tail)==2 and 'a'<=tail[0]<='z' and tail[1] in '123456789':order.append((ord(tail[0])-97)*9+int(tail[1])-1)
    n=max(order,default=-1)+1;require(n<234,'date version suffix exhausted')
    return directory/f'{stem}_{date}{chr(97+n//9)}{1+n%9}.json'

def prepare(root,config):
    cfg,roster=load_config(config);require(cfg.get('budget_parent'),'C2 requires explicit closed C1 budget parent');require(root==Path(cfg['budget_run_root']).resolve(),'C2 only approved production root; no budget reset roots');ledger=Ledger(root,cfg);ledger.check_time(model=True);ledger.assert_no_unknown()
    code=code_identity()
    for v in cfg['assets'].values():verify_ref(v)
    require(Path(sys.executable).resolve()==Path('/Users/zhuanz/.venvs/openwebui/bin/python').resolve(),'use frozen Python environment')
    audit=audit_set(roster);manifest_path=root/'run_manifest.json';writer=Writer(ledger)
    if manifest_path.exists():
        man=load(manifest_path);require(man['config']==ref(config) and man['code']==code and man['audit']==audit,'existing run identity changed')
    else:
        require(not list((root/'episode_runs').glob('*')),'unmanifested experiment directory')
        man={'schema':'r11_run_v1','task':'T011.08','created_at':now(),'run_id':root.name,'output':str(root),'python':sys.executable,'cwd':str(P),'code':code,'config':ref(config),'budget_parent':cfg.get('budget_parent'),'numeric_hash':numerical_hash(cfg),'scheduling_hash':digest({'deadlines':cfg['deadlines'],'limits':cfg['call_limits'],'storage':cfg['storage_limits']}),'audit':audit,'spec':cfg['spec'],'adoption':cfg['adoption'],'kit_source_hashes':{str(p):sha(p) for p in Path(cfg['kit']).rglob('*.py')},'source_reports':[ref(Path(cfg['root'])/'DD/调研'/name) for name in ('DD_调研_Q011.01_20261017a1.md','DD_调研_Q011.02_20261017.md','DD_调研_Q011.03_20261017a1.md')],'old_producer_P':'d73f2cd7ea6f8f97093558f820884ae4d74ee0bf','tracker_producer':'eeaa540ca1a0d46134e04fca76ab8c1205ffca5a','old_P_cache_import':'disabled','provenance_kind':'local_git','submission_eligible':False}
        writer.json(manifest_path,man,exclusive=True);writer.json(root/'audit_frozen.json',audit,exclusive=True)
    inputs=[]
    for row in roster['episodes']:
        path=root/'inputs'/f'episode_{row["episode"]:06d}.json'
        if path.exists():verify_episode(load(path),cfg);inputs.append(ref(path));continue
        ep=freeze_episode(row,cfg)
        if ep is not None:writer.json(path,ep,exclusive=True);inputs.append(ref(path))
    snapshot={'schema':'r11_data_snapshot_v1','as_of':now(),'manifest':cfg['assets']['roster'],'episodes':inputs,'pending':[r['episode'] for r in roster['episodes'] if not (root/'inputs'/f'episode_{r["episode"]:06d}.json').exists()]}
    snap=next_dated(root/'data_snapshots','data_snapshot',cfg['date'].replace('-',''));writer.json(snap,snapshot,exclusive=True);writer.json(root/'data_snapshot_latest.json',ref(snap))
    return evidence(root,cfg,man)

def stage_receipt(er,stage):
    path=er/f'{stage}_latest.json'
    if not path.exists():return None
    rr=load(path);verify_ref(rr);return load(rr['path'])

def evidence(root,cfg,man):
    episodes=[];complete={};errors=[]
    for p in sorted((root/'inputs').glob('episode_*.json')):
        ep=load(p);er=root/'episode_runs'/p.stem;row={'episode':ep['episode'],'input':ref(p),'source':{'mirror_declaration_match':ep['mirror_declaration_match'],'official_independent_binding':'passed' if ep['official_independent_binding'] else 'pending','local_hash':'passed','wholePTS':'passed'},'required':ep['required'],'stages':{}}
        try:
            records=FrameStore(er,Writer(Ledger(root,cfg))).records();row['detector_done']=sorted(records);row['hmr_valid']=sorted(f for f,r in records.items() if r['hmr_outcome']=='hmr_valid');row['hmr_failed']=sorted(f for f,r in records.items() if r['hmr_outcome']=='hmr_failed');row['remaining_A']=sorted(set(ep['required'])-records.keys())
            for stage in ('A','B','C'):
                receipt=stage_receipt(er,stage)
                row['stages'][stage]={'status':receipt['status'] if receipt else 'pending','receipt':load(er/f'{stage}_latest.json') if receipt else None}
                if stage=='B' and receipt and receipt['status']=='completed':row['rig_checked']=receipt['details']['rig_checked'];row['hmr_filled']=receipt['details']['hmr_filled']
                if stage=='C' and receipt and receipt['status']=='completed':
                    verify_ref(receipt['details']['pack']);pack=load(receipt['details']['pack']['path']);verify_ref(pack['candidate']);row['pack']=receipt['details']['pack'];row['pack_verified']=ep['required'];complete[ep['episode']]=ep['required']
            if ep['episode'] not in complete and (er/'C_numeric_committed.json').exists():
                nr=load(er/'C_numeric_committed.json');verify_ref(nr['pack']);pack=load(nr['pack']['path']);verify_ref(pack['candidate']);row['pack']=nr['pack'];row['pack_verified']=ep['required'];complete[ep['episode']]=ep['required']
            row['runtime']={stage:ref(er/f'runtime_{stage}.json') for stage in ('A','B') if (er/f'runtime_{stage}.json').exists()}
            row['completed_sources']=[i for i,source in enumerate(ep['sources']) if set(source)<=set(records)]
            if (er/'B_plan_latest.json').exists():
                pr=load(er/'B_plan_latest.json');verify_ref(pr);plan=load(pr['path']);br=Path(pr['path']).parent;checked=[]
                for p in sorted((br/'frames').glob('*.json')):
                    rr=load(p);require(rr['plan_hash']==pr['plan_hash'],'partial rig plan mismatch');verify_ref(rr['output']);checked.append(rr['frame'])
                row['rig_checked']=checked;row['hmr_filled']=[int(f) for f,parent in plan['parents'].items() if int(f)!=parent['frame']]
            row['quality']='pending_human_review_contact_real_scale_surface_unknown'
        except Exception as exc:row['evidence_error']=repr(exc);errors.append({'episode':ep['episode'],'error':repr(exc)})
        episodes.append(row)
    agg=root/'aggregate_latest.json';aggregate=None
    if agg.exists():
        try:verify_ref(load(agg));aggregate=load(agg)
        except Exception as e:errors.append({'aggregate':repr(e)})
    roster=load(cfg['assets']['roster']['path']);state=Ledger(root,cfg).snapshot()
    result={'schema':'r11_evidence_index_v1','as_of':now(),'date':cfg['date'],'run_id':root.name,'run_manifest':ref(root/'run_manifest.json'),'code':{'commit':man['code']['commit'],'tree':man['code']['tree']},'spec':cfg['spec'],'adoption':cfg['adoption'],'data_snapshot':load(root/'data_snapshot_latest.json') if (root/'data_snapshot_latest.json').exists() else None,'audit_frozen':ref(root/'audit_frozen.json'),'episodes':episodes,'completed_episodes':sorted(complete),'remaining_episodes':[r['episode'] for r in roster['episodes'] if r['episode'] not in complete],'completed_frame_count':sum(map(len,complete.values())),'required_frame_count':9877,'aggregate':aggregate,'cost':state,'budget_accounting':state['accounting'],'budget_parent':cfg.get('budget_parent'),'load_counts':{k+'_count':state['counts'].get(k,0) for k in ('dnn_load','ort_session_load','rig_load')},'unknown_attempts':list(state['pending'].values()),'errors':errors,'states':{'implementation':'frozen_local_source','numerical':'see_per_episode_receipts','coverage':'full' if len(complete)==30 and sum(map(len,complete.values()))==9877 else 'partial','quality':'pending','real_scale_contact':'unknown','numeric_index_mesh_pack':'see_pack_receipts','official_writer':'not_used_original_terminal_write_local_L1','official_code_commit_url_format':'failed_local_marker_expected','remote_commit_accessibility':'unpublished','kaggle_scoring':'not_run','public_row':'not_run','website_record':'not_run'},'provenance_kind':'local_git','submission_eligible':False,'runtime_hard_stop':cfg['deadlines']['hard_stop']}
    snap=next_dated(root/'evidence_snapshots','evidence_index',cfg['date'].replace('-',''));atomic_json(snap,result,True);atomic_json(root/'evidence_index.json',{**result,'snapshot':ref(snap)})
    return {'evidence_index':str(root/'evidence_index.json'),'snapshot':ref(snap),'complete_episodes':sorted(complete),'completed_frames':sum(map(len,complete.values())),'errors':errors}

def global_new_bytes(cfg):
    cutoff=datetime.datetime.fromisoformat(cfg['window_start']).timestamp();total=0
    # Infrequent conservative full-DD reconciliation includes concurrent T05/T07/T90/controller outputs.
    for directory,dirs,files in os.walk(Path(cfg['root'])/'DD',followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink() and d!='__pycache__']
        for name in files:
            path=Path(directory)/name
            try:
                st=path.stat()
                if st.st_mtime>=cutoff:total+=st.st_size
            except FileNotFoundError:pass
    return total

def supervise(root,cfg,man,stage,ep):
    import psutil
    stagekey='C' if stage=='aggregate' else stage;ledger=Ledger(root,cfg);ledger.check_time(model=True);ledger.assert_no_unknown();verify_code(man)
    for p,h in man['kit_source_hashes'].items():require(sha(p)==h,'kit changed')
    limits=cfg['stage_limits'][stagekey];directory=root/'supervision';directory.mkdir(parents=True,exist_ok=True);aid=__import__('uuid').uuid4().hex
    logprefix=directory/f'{stage}_{ep}_{aid}';started=time.monotonic();lastscan=0;peak=0;reason=None;newbytes=global_new_bytes(cfg)
    if newbytes>=cfg['global_new_file_limit']:
        with ledger.edit() as state:state['stop_reason']='global_new_file_limit'
        raise BudgetStop('global_new_file_limit')
    with ledger.edit() as state:state['global_observation']={'global_bytes':newbytes,'local_accounted_bytes':sum(state['written_bytes'].values()),'as_of':now()}
    if stage=='aggregate':cmd=[sys.executable,'-B',str(P/'r11.py'),'--stage','aggregate-worker','--output',str(root)]
    else:cmd=[sys.executable,'-B',str(P/'r11_worker.py'),'--stage',stage,'--root',str(root),'--episode',str(ep)]
    env=os.environ.copy();env.update({'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'})
    def interrupt(*_):raise BudgetStop('supervisor_interrupt')
    oldterm=signal.signal(signal.SIGTERM,interrupt);oldint=signal.signal(signal.SIGINT,interrupt);proc=None
    try:
        with logprefix.with_suffix('.stdout.log').open('xb') as stdout,logprefix.with_suffix('.stderr.log').open('xb') as stderr,logprefix.with_suffix('.resources.jsonl').open('xb') as monitor:
            proc=subprocess.Popen(cmd,cwd=P,env=env,stdout=stdout,stderr=stderr,start_new_session=True);pp=psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    elapsed=time.monotonic()-started;resident=pp.memory_info().rss
                    for c in pp.children(recursive=True):
                        try:resident+=c.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    peak=max(peak,resident);ledger.check_time(model=True)
                    if elapsed>limits['wall_s']:raise BudgetStop('stage_wall_limit')
                    if resident>limits['rss_bytes']:raise BudgetStop('process_tree_rss_limit')
                    if elapsed-lastscan>=cfg['global_scan_interval_s']:
                        newbytes=global_new_bytes(cfg);lastscan=elapsed
                        with ledger.edit() as state:state['global_observation']={'global_bytes':newbytes,'local_accounted_bytes':sum(state['written_bytes'].values()),'as_of':now()}
                        if newbytes>=cfg['global_new_file_limit']:raise BudgetStop('global_new_file_limit')
                    monitor.write(json_bytes({'elapsed_s':elapsed,'tree_rss_bytes':resident,'sampled_peak_rss_bytes':peak,'global_new_bytes_last_reconciliation':newbytes})+b'\n');monitor.flush()
                    time.sleep(cfg['monitor_interval_s'])
                except psutil.NoSuchProcess:break
    except BaseException as exc:
        reason=repr(exc)
        if isinstance(exc,BudgetStop) and any(k in str(exc) for k in ('limit','hard_stop','candidate_stop','headroom')):
            with ledger.edit() as state:state['stop_reason']=reason
        if proc and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=min(3,max(.1,datetime.datetime.fromisoformat(cfg['deadlines']['candidate_freeze']).timestamp()-time.time())))
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait(timeout=3)
        raise
    finally:
        signal.signal(signal.SIGTERM,oldterm);signal.signal(signal.SIGINT,oldint)
        if proc and proc.poll() is None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait(timeout=3)
        atomic_json(logprefix.with_suffix('.json'),{'command':cmd,'cwd':str(P),'python':sys.executable,'commit':man['code']['commit'],'stage':stage,'episode':ep,'returncode':proc.returncode if proc else None,'reason':reason,'elapsed_s':time.monotonic()-started,'sampled_peak_tree_rss_bytes':peak,'global_new_bytes':newbytes,'finished_at':now()},True)
    require(proc.returncode==0,'stage failed; see '+str(logprefix.with_suffix('.stderr.log')))
    verify_code(man)
    for p,h in man['kit_source_hashes'].items():require(sha(p)==h,'kit changed during stage')

def validate_completed(root,ep,cfg):
    er=root/'episode_runs'/f'episode_{ep["episode"]:06d}';rr=stage_receipt(er,'C')
    if not rr or rr['status']!='completed':
        if not (er/'C_numeric_committed.json').exists():return False
        numeric=load(er/'C_numeric_committed.json');rr={'details':numeric}
    pack=load(verify_ref(rr['details']['pack']));verify_ref(pack['candidate']);require(pack['numeric_index_mesh_pack']=='passed' and pack['completed'][str(ep['episode'])]==ep['required'],'completed pack identity mismatch')
    for r in rr['details']['inputs'].values():
        if isinstance(r,dict) and 'path' in r and 'sha256' in r:verify_ref(r)
    return True

def run(root,cfg,man,episodes):
    ledger=Ledger(root,cfg);ledger.assert_no_unknown()
    with Owner(root/'supervisor_owner','run',None):
        for e in episodes:
            ledger.assert_no_unknown()
            inp=root/'inputs'/f'episode_{e:06d}.json';require(inp.exists(),'episode data not frozen: '+str(e));ep=load(inp);verify_episode(ep,cfg)
            if validate_completed(root,ep,cfg):continue
            er=root/'episode_runs'/inp.stem
            if e!=16:
                first=root/'inputs/episode_000016.json';require(first.exists() and validate_completed(root,load(first),cfg),'first240 must pass before other episodes')
            # Once an episode model stage started, it may finish before 04:50; 04:40 blocks a fresh one.
            prior=ledger.snapshot()['episode_counts']
            if not any(prior.get(str(e)+':'+k,0) for k in ('dnn_load','ort_session_load','rig_load')):ledger.check_time(new_episode=True,model=True)
            for stage in ('A','B','C'):
                ledger.assert_no_unknown()
                r=stage_receipt(er,stage)
                if r and r['status']=='completed':
                    if stage=='A':
                        verify_ref(r['details']['frame_journal']);records=FrameStore(er,Writer(ledger)).records();require(set(records)==set(ep['required']),'A coverage mismatch')
                        for record in records.values():
                            if record.get('raw'):verify_ref(record['raw'])
                    elif stage=='B':
                        for k in ('plan','identity','mesh'):verify_ref(r['details'][k])
                    continue
                try:supervise(root,cfg,man,stage,e)
                finally:evidence(root,cfg,man)
    ledger.assert_no_unknown()
    return evidence(root,cfg,man)

def aggregate_worker(root,cfg,man):
    from r11_pack import pack_local,check_dense
    ledger=Ledger(root,cfg);ledger.assert_no_unknown();writer=Writer(ledger);coverage={};inputs={}
    with Owner(root/'aggregate_owner','aggregate',None):
        for p in sorted((root/'inputs').glob('episode_*.json')):
            ep=load(p)
            if not validate_completed(root,ep,cfg):continue
            er=root/'episode_runs'/p.stem;r=stage_receipt(er,'C');entry=r['details']['inputs'] if r and r['status']=='completed' else load(er/'C_numeric_committed.json')['inputs'];data=read_npz(verify_ref(entry['npz']));mask=read_npz(verify_ref(entry['mask']))['mask'];check_dense(data,ep['required'],mask)
            coverage[ep['episode']]=ep['required'];inputs[ep['episode']]=entry
        require(coverage,'no completed episode candidate');label='aggregate_'+digest({str(k):v for k,v in inputs.items()})[:16]
        result=pack_local(root,cfg,man,coverage,inputs,writer,label);ledger.assert_no_unknown();writer.json(root/'aggregate_latest.json',ref(root/'packs'/label/'validation.json'))
        return {'candidate':result['candidate'],'rows':result['rows'],'coverage':result['coverage']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','run','aggregate','status','aggregate-worker'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--config',type=Path,default=DEFAULT);ap.add_argument('--episodes',default='16');args=ap.parse_args();root=args.output.resolve()
    if args.stage=='prepare':
        preflight,_=load_config(args.config.resolve());require(preflight.get('budget_parent') and root==Path(preflight['budget_run_root']).resolve(),'C2 only approved production root')
        parent=load(verify_ref(preflight['budget_parent']));require(root!=Path(parent['parent_run']).resolve(),'C1 root is read only')
        with Owner(root/'supervisor_owner','prepare',None):result=prepare(root,args.config.resolve())
    else:
        man=load(root/'run_manifest.json');verify_ref(man['config']);cfg,roster=load_config(man['config']['path']);require(root==Path(cfg['budget_run_root']).resolve(),'C2 only approved root');verify_code(man)
        if args.stage=='status':result=evidence(root,cfg,man)
        elif args.stage=='run':
            episodes=[16]+[r['episode'] for r in roster['episodes'] if r['episode']!=16] if args.episodes=='all' else [r['episode'] for r in roster['episodes'] if r['episode']!=16] if args.episodes=='remaining' else [int(x) for x in args.episodes.split(',')]
            require(len(episodes)==len(set(episodes)) and set(episodes)<=set(range(30)),'invalid episode selection');result=run(root,cfg,man,episodes)
        elif args.stage=='aggregate-worker':result=aggregate_worker(root,cfg,man)
        else:
            with Owner(root/'supervisor_owner','aggregate',None):
                try:supervise(root,cfg,man,'aggregate',None)
                finally:result=evidence(root,cfg,man)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
