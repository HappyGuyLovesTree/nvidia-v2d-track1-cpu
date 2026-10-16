"""Explicit bounded sessions and a single project-wide production owner. No model I/O."""
from pathlib import Path
import datetime, fcntl, json, math, os, shutil, uuid
import psutil
from common import load
from tracking import digest
from r11_state import TZ, now, require, ref, verify_ref, atomic_json, Ledger
from r11_inputs import code_identity, verify_code
from r12_supervision import read_attempt, Policy

SESSION='r12_execution_session_v1'
CLEANUP={'exit_grace_s':5.,'term_wait_s':5.,'kill_wait_s':5.,'cleanup_total_s':20.,
         'telemetry_retry_count':2,'telemetry_retry_total_s':1.}

def owner_observation(record):
    pid=record['pid'];started=record.get('pid_started',record.get('create_time'))
    require(type(pid) is int and pid>0 and isinstance(started,(int,float)) and started>0,'unknown owner identity')
    try:
        p=psutil.Process(pid);current=p.create_time()
        require(current>0,'unknown live creation time')
        if abs(current-started)>.001:return {'status':'pid_reused','observed_at':now(),'current_create_time':current}
        # Zombies still require explicit independent handling; do not call gone.
        raise ValueError('registered same-identity owner still present: '+str(pid))
    except psutil.NoSuchProcess:return {'status':'gone','observed_at':now()}

def history_owners(roots):
    result=[]
    for root in roots:
        for path in sorted(Path(root).rglob('owner_latest.json')):
            r=load(path);end=path.parent/'attempts'/f"{r['attempt']}_end.json"
            result.append({'owner_ref':ref(path),'identity':r,'end_ref':ref(end) if end.exists() else None,
                           'live_observation':owner_observation(r)})
    return result

def validate_handover(handover_ref,roots):
    value=load(verify_ref(handover_ref));require(value['schema']=='r12_historical_owner_handover_v1','wrong handover')
    require(value['roots']==[str(Path(p).resolve()) for p in roots],'handover roots mismatch')
    expected={str(p.resolve()) for root in roots for p in Path(root).rglob('owner_latest.json')}
    require({r['owner_ref']['path'] for r in value['owners']}==expected,'historical owner set changed')
    for row in value['owners']:
        require(load(verify_ref(row['owner_ref']))==row['identity'],'owner identity changed')
        if row['end_ref']:verify_ref(row['end_ref'])
        owner_observation(row['identity'])
    return value

def validate_session(session,root,*,check_time=True):
    require(session['schema']==SESSION and session['mode']=='production_resume','fresh mode forbidden in production entry')
    require(Path(session['active_root']).resolve()==Path(root).resolve(),'session/root mismatch')
    for r in session['authorization_refs']:verify_ref(r)
    for key in ('parent_bundle_ref','parent_index_ref','historical_handover_ref','old_new_diff_ref'):verify_ref(session[key])
    require(session['original_window_start']=='2026-10-17T01:06:03+08:00','global baseline reset forbidden')
    times=[datetime.datetime.fromisoformat(session[k]) for k in ('start_at','new_episode_stop','candidate_stop','freeze_at','hard_stop')]
    require(all(t.tzinfo is not None for t in times),'timezone required')
    a,b,c,d,e=times;require(a<b<=c<=d<e,'invalid session time order')
    policy=Policy.from_session(session)
    require((e-d).total_seconds()>=policy.cleanup_total_s and (d-c).total_seconds()>=policy.cleanup_total_s,'insufficient freeze/cleanup headroom')
    require(session['controller_code']==session['engine_code'],'unvalidated controller/engine split')
    if check_time:require(a<=datetime.datetime.now(TZ)<e,'session not started or expired')
    return session

def runtime_config(man,root):
    cfg=load(verify_ref(man['config']))
    if man.get('schema')!='r12_run_v1':return cfg
    session=load(verify_ref(man['session_ref']));validate_session(session,root)
    require(session['controller_code']==man['code'],'session/code mismatch')
    require(session['call_caps']==cfg['call_limits'] and session['storage_caps']==cfg['storage_limits'] and session['stage_limits']==cfg['stage_limits'],'resource cap change forbidden')
    require(session['global_occupancy_policy']=={'root':str(Path(cfg['root'])/'DD'),'window_start':cfg['window_start'],'cap':cfg['global_new_file_limit']},'global resource reset forbidden')
    cfg={**cfg,'execution_session':{**session,'session_ref':man['session_ref']},
         'budget_parent':session['parent_bundle_ref'],'budget_run_root':str(Path(root).resolve()),
         'deadlines':{'new_episode_stop':session['new_episode_stop'],'candidate_stop':session['candidate_stop'],
                      'candidate_freeze':session['freeze_at'],'hard_stop':session['hard_stop']}}
    return cfg

def global_directory(cfg):return Path(cfg['root'])/'DD/生产/track1/r12_control'

def check_registry(directory,root,parent_ref):
    p=Path(directory)/'registry.json'
    if p.exists():
        r=load(p);require(r['active_root']==str(Path(root).resolve()) and r['parent_bundle_ref']==parent_ref,'cross-root budget reset refused; reuse registered root')
    else:raise ValueError('production root is not registered')

def assert_closed(root,exclude=None):
    for p in Path(root).rglob('*.started.json'):
        if exclude and p.resolve()==Path(exclude).resolve():continue
        value=read_attempt(p)
        require(value.get('supervisor_outcome')=='passed','unresolved supervision/controller: '+str(p))
    for p in Path(root).rglob('owner_latest.json'):
        r=load(p);require((p.parent/'attempts'/f"{r['attempt']}_end.json").exists(),'unclosed local owner: '+str(p))

def child_bridge(root,cfg,episode=None,stage=None):
    session=cfg['execution_session'];directory=global_directory(cfg)
    check_registry(directory,root,cfg['budget_parent'])
    claim=load(directory/'owner_latest.json')
    require(claim['session_id']==session['session_id'] and claim['active_root']==str(Path(root).resolve()),'worker global owner mismatch')
    p=psutil.Process(claim['pid']);require(abs(p.create_time()-claim['pid_started'])<.001,'controller identity changed')
    require(any(a.pid==p.pid and abs(a.create_time()-p.create_time())<.001 for a in psutil.Process().parents()),'worker not descendant of registered controller')
    require(not (directory/'attempts'/f"{claim['attempt']}_end.json").exists(),'controller already ended')
    from r12 import release_gate
    man=load(Path(root)/'run_manifest.json')
    release_gate(Path(root),man,cfg,[episode] if episode is not None else list(range(23,30)))
    require(('controller_run_' in claim['controller_prefix']) if episode is not None else ('controller_aggregate_' in claim['controller_prefix']),'worker/controller operation mismatch')
    require(stage in ('A','B','C') if episode is not None else stage=='aggregate','worker stage missing')
    return claim

class GlobalOwner:
    def __init__(self,root,cfg,prefix):self.root=Path(root);self.cfg=cfg;self.directory=global_directory(cfg);self.prefix=Path(prefix)
    def __enter__(self):
        self.directory.mkdir(parents=True,exist_ok=True);self.fd=(self.directory/'owner.lock').open('a+')
        try:
            fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            session=self.cfg['execution_session'];validate_session(session,self.root)
            check_registry(self.directory,self.root,self.cfg['budget_parent'])
            validate_handover(session['historical_handover_ref'],session['historical_roots'])
            from r12_budget import validate_current, validate_chain
            validate_chain(self.cfg['budget_parent'])
            validate_current(self.root,self.cfg)
            assert_closed(self.root,self.prefix.with_suffix('.started.json'))
            latest=self.directory/'owner_latest.json';prev=load(latest) if latest.exists() else None
            if prev:
                owner_observation(prev)
                require((self.directory/'attempts'/f"{prev['attempt']}_end.json").exists(),'global owner missing end; independent handover required')
            p=psutil.Process();self.attempt=uuid.uuid4().hex
            self.record={'attempt':self.attempt,'parent_attempt':prev['attempt'] if prev else None,'pid':p.pid,'pid_started':p.create_time(),
                         'uid':os.getuid(),'ppid':p.ppid(),'pgid':os.getpgid(0),'sid':os.getsid(0),'started_at':now(),
                         'session_id':session['session_id'],'active_root':str(self.root.resolve()),
                         'controller_prefix':str(self.prefix.resolve())}
            atomic_json(self.directory/'attempts'/f'{self.attempt}.json',self.record,True);atomic_json(latest,self.record)
            return self
        except BaseException:
            fcntl.flock(self.fd,fcntl.LOCK_UN);self.fd.close();raise
    def __exit__(self,typ,val,tb):
        try:atomic_json(self.directory/'attempts'/f'{self.attempt}_end.json',{'attempt':self.attempt,'finished_at':now(),'outcome':'completed' if typ is None else 'failed','error':None if val is None else repr(val)},True)
        finally:fcntl.flock(self.fd,fcntl.LOCK_UN);self.fd.close()

def make_session(root,man,parent_ref,handover_ref,authorization,diff_ref,*,minutes=150,release_ref=None):
    require(type(minutes) is int and 10<=minutes<=360,'bounded 10..360 minute session required')
    cfg=load(verify_ref(man['config']));code=code_identity();require(code==man['code'],'frozen run code changed')
    parent=load(verify_ref(parent_ref));c1=load(verify_ref(parent['C1_bundle_ref']))
    roots=[c1['parent_run'],load(verify_ref(parent['parent_index_ref']))['root']]
    validate_handover(handover_ref,roots)
    start=datetime.datetime.now(TZ);end=start+datetime.timedelta(minutes=minutes)
    session={'schema':SESSION,'mode':'production_resume','session_id':uuid.uuid4().hex,'created_at':now(),
      'active_root':str(Path(root).resolve()),'authorization_refs':[authorization], 'controller_code':code,'engine_code':code,
      'parent_bundle_ref':parent_ref,'parent_index_ref':parent['parent_index_ref'],'historical_handover_ref':handover_ref,'historical_roots':roots,
      'original_window_start':cfg['window_start'],'start_at':start.isoformat(),'new_episode_stop':(end-datetime.timedelta(minutes=10)).isoformat(),
      'candidate_stop':(end-datetime.timedelta(seconds=90)).isoformat(),'freeze_at':(end-datetime.timedelta(seconds=45)).isoformat(),
      'hard_stop':end.isoformat(),'cleanup':CLEANUP,'call_caps':cfg['call_limits'],'storage_caps':cfg['storage_limits'],'stage_limits':cfg['stage_limits'],
      'global_occupancy_policy':{'root':str(Path(cfg['root'])/'DD'),'window_start':cfg['window_start'],'cap':cfg['global_new_file_limit']},
      'change_allowlist':['supervision','session','budget','import','export','worker_entry_owner_bridge'],
      'old_new_diff_ref':diff_ref,'production_release_ref':release_ref,'numerical_recipe_commit':parent['parent_history'].get('numerical_recipe_commit','483f8d61690d7df229c57dca43750b23afcfbf57')}
    validate_session(session,root);return session
