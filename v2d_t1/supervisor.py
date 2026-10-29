"""Production supervisor. Persistent one-shot calls, sampled process-tree resources."""
import argparse,contextlib,datetime,fcntl,json,os,re,signal,subprocess,sys,time,uuid
from pathlib import Path
from .core import encoded, write_new, reference, digest, verify
from .core import need, ContractError

RESOURCE_POLICY=dict(seconds=1200,output_bytes=1073741824,rss_bytes=4294967296,
    rss_period_s=1,system_period_s=10,rss_gap_s=3,system_unknown_s=20,
    critical_consecutive=2,hmr=24,rig=24,workers=1)

def append(path,event):
    event=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),monotonic=time.monotonic(),**event)
    with Path(path).open('ab') as f:f.write(encoded_line(event));f.flush();os.fsync(f.fileno())
def encoded_line(e):return (json.dumps(e,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
def clock():return dict(wall=time.time(),monotonic=time.monotonic())
def process_identity(pid):
    import psutil
    p=psutil.Process(pid);return dict(pid=pid,created=p.create_time(),ppid=p.ppid(),executable=p.exe())
def rss_observation(pid,identity=None,extra_pid=None):
    import psutil
    out=dict(value=None,pids=[],error=None,source='psutil RSS bytes; supervisor plus worker and recursive descendants')
    try:
        p=psutil.Process(pid)
        need(identity is None or p.create_time()==identity['created'],'root PID reused')
        processes=[p]+p.children(recursive=True)
        if extra_pid is not None:
            x=psutil.Process(extra_pid);processes += [x]+x.children(recursive=True)
        unique={x.pid:x for x in processes};out['pids']=[process_identity(i) for i in sorted(unique)]
        out['value']=sum(x.memory_info().rss for x in unique.values())
    except Exception as e:out['error']=type(e).__name__+': '+str(e)
    return out

def system_observation():
    out=dict(pressure=dict(value=None,raw=None,error=None,source='sysctl kern.memorystatus_vm_pressure_level; 1 normal / 2 warning / 4 critical'),
             swap=dict(value=None,raw=None,error=None,source='sysctl vm.swapusage; used bytes'))
    for name,key in [('pressure','kern.memorystatus_vm_pressure_level'),('swap','vm.swapusage')]:
        e=out[name]
        try:
            cp=subprocess.run(['/usr/sbin/sysctl','-n',key],capture_output=True,text=True,timeout=2)
            e.update(raw=cp.stdout,stderr=cp.stderr,exit=cp.returncode)
            need(cp.returncode==0,'sysctl exit '+str(cp.returncode))
            if name=='pressure':
                need(cp.stdout.strip() in ('1','2','4'),'unrecognized pressure level');e['value']=int(cp.stdout.strip())
            else:
                m=re.search(r'used\s*=\s*([0-9.]+)([KMGT]?)',cp.stdout);need(m is not None,'unrecognized swap usage')
                e['value']=int(float(m[1])*1024**('KMGT'.index(m[2])+1 if m[2] else 0))
        except Exception as ex:e['error']=type(ex).__name__+': '+str(ex)
    return out

def byte_snapshot(root):
    files={}
    for p in Path(root).rglob('*'):
        need(not p.is_symlink(),'output symlink')
        if p.is_file():files[str(p.relative_to(root))]=p.stat().st_size
    return files

class StopState:
    """Pure stop decision; tested with injected observations and monotonic times."""
    def __init__(self,policy):
        self.policy=policy;self.last_rss=None;self.system_failed_since=None;self.critical=0;self.max_rss=None;self.last_system=None
    def check(self,elapsed,rss,system,total_bytes):
        if elapsed>=self.policy['seconds']:return 'machine_time_limit'
        if total_bytes>=self.policy['output_bytes']:return 'output_byte_limit'
        if rss['error'] is None and rss['value'] is not None:
            self.last_rss=elapsed;self.max_rss=max(self.max_rss or 0,rss['value'])
            if rss['value']>=self.policy['rss_bytes']:return 'sampled_tree_RSS_limit'
        if self.last_rss is None:self.last_rss=0
        if elapsed-self.last_rss>self.policy['rss_gap_s']:return 'RSS_observation_gap'
        if system is not None:
            self.last_system=system
            bad=any(system[k]['error'] is not None or system[k]['value'] is None for k in ('pressure','swap'))
            if bad:
                if self.system_failed_since is None:self.system_failed_since=elapsed
            else:self.system_failed_since=None
            self.critical=self.critical+1 if system['pressure']['value']==4 and system['pressure']['error'] is None else 0
            if self.critical>=self.policy['critical_consecutive']:return 'consecutive_critical_pressure'
        if self.system_failed_since is not None and elapsed-self.system_failed_since>self.policy['system_unknown_s']:return 'system_observation_unknown'
        return None

class Calls:
    """Worker call journal. Start persists before load/forward; no retries/duplicate keys."""
    def __init__(self,root,limits):
        self.path=Path(root)/'calls.jsonl';self.limits=limits
        if not self.path.exists():write_new(self.path,b'')
    def rows(self):
        raw=self.path.read_bytes();need(not raw or raw.endswith(b'\n'),'torn call journal')
        return [json.loads(x) for x in raw.splitlines()]
    @contextlib.contextmanager
    def call(self,kind,key,parents):
        rows=self.rows();starts=[r for r in rows if r['event']=='start'];ends={r['call_id'] for r in rows if r['event'] in ('end','failed')}
        need(all(s['call_id'] in ends for s in starts),'unclosed model call; investigation required')
        need(not any(s['kind']==kind and s['key']==key for s in starts),'duplicate/retry call denied')
        if kind in ('hmr','rig','hmr_load','rig_load'):need(sum(s['kind']==kind for s in starts)<self.limits.get(kind,1),kind+' call limit')
        call_id=uuid.uuid4().hex;begin=clock();identity=process_identity(os.getpid())
        append(self.path,dict(event='start',kind=kind,key=key,parents=parents,call_id=call_id,process=identity,start=begin))
        token=dict(call_id=call_id)
        try:
            yield token
            if token.get('output') is not None:verify(token['output'])
        except BaseException as e:
            append(self.path,dict(event='failed',kind=kind,key=key,call_id=call_id,elapsed=time.monotonic()-begin['monotonic'],error=type(e).__name__+': '+str(e)[:500]));raise
        else:
            append(self.path,dict(event='end',kind=kind,key=key,call_id=call_id,elapsed=time.monotonic()-begin['monotonic'],output=token.get('output'),output_identity=token.get('output_identity')))

def journal_summary(path):
    raw=Path(path).read_bytes();need(not raw or raw.endswith(b'\n'),'torn parent journal')
    rows=[json.loads(x) for x in raw.splitlines()];starts=[r for r in rows if r['event']=='start']
    ends={r['call_id']:r for r in rows if r['event'] in ('end','failed')}
    need(all(s['call_id'] in ends for s in starts),'unknown parent call cannot be treated as zero')
    return {k:sum(s['kind']==k for s in starts) for k in ('hmr','rig','hmr_load','rig_load')}

def supervise(root,argv,policy=None,*,authority,fixture=False,resume=None,initial_carry=None):
    root=Path(root).resolve();p=dict(RESOURCE_POLICY if policy is None else policy)
    need(root.is_absolute() and root.exists() and p['workers']==1,'supervisor root/worker count')
    with (root/'run.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        carry_seconds=0.;carry_bytes=0;parent=None;attempt='initial'
        if initial_carry is not None:
            need(resume is None,'initial carry cannot replace resume')
            need(initial_carry['seconds']>=0 and initial_carry['bytes']>=0,'nonnegative preparation/failure carry')
            for r in initial_carry['evidence']:verify(r)
            for r in initial_carry.get('previous_receipts',[]):
                old=json.loads(verify(r).read_text());need(old['status']=='stopped' and old['exit'] is not None and old['reap'] is not None,'previous full attempt must be stopped/reaped')
                need(initial_carry['seconds']>=old['elapsed'] and initial_carry['bytes']>=old['new_bytes'],'full carry cannot decrease')
            carry_seconds=initial_carry['seconds'];carry_bytes=initial_carry['bytes']
        if resume is not None:
            need(resume['schema']=='r020_single_recovery_v1' and resume['execution_root']==str(root),'explicit same-root recovery contract')
            need(resume['authorized'] is True and resume['extra_ort_loads']==1 and resume['automatic_retry'] is False,'single explicit recovery authority')
            verify(resume['authorization']);verify(resume['parent_receipt']);verify(resume['parent_origin']);verify(resume['parent_call_journal'])
            parent=json.loads(Path(resume['parent_receipt']['path']).read_text());old_origin=json.loads(Path(resume['parent_origin']['path']).read_text())
            need(parent['status']=='stopped' and parent['exit'] is not None and parent['reap'] is not None,'parent process not reaped')
            need(resume['carry_seconds']==parent['elapsed'] and resume['carry_calls']==journal_summary(resume['parent_call_journal']['path']),'carry mismatch')
            need(old_origin['policy']==p and old_origin['run_id']==parent['run_id'],'same original run/policy required')
            old_child=json.loads((root/'child_identity.json').read_text())
            import psutil
            try:need(psutil.Process(old_child['pid']).create_time()!=old_child['created'],'parent worker still running')
            except psutil.NoSuchProcess:pass
            carry_seconds=resume['carry_seconds'];carry_bytes=resume['carry_bytes'];run_id=parent['run_id'];attempt='resume_01'
            need(carry_bytes>=parent['new_bytes'],'output carry cannot decrease')
        else:run_id=uuid.uuid4().hex
        attempt_origin=dict(run_id=run_id,attempt=attempt,start=clock(),supervisor=process_identity(os.getpid()),argv=argv,policy=p,authority=authority,fixture=fixture,carry_seconds=carry_seconds,carry_bytes=carry_bytes)
        write_new(root/('origin.json' if resume is None else 'origin_resume_01.json'),attempt_origin,True)
        high=byte_snapshot(root);total=carry_bytes+(sum(high.values()) if resume is None else 0);st=StopState(p);st.last_rss=carry_seconds
        if parent is not None:st.max_rss=parent['sampled_tree_rss_max_bytes']
        child=None;reason=None;cp_exit=None;last=0;last_system=-1e9;reap=None
        resources=root/('resources.jsonl' if resume is None else 'resources_resume_01.jsonl')
        stopfile=root/('stop.json' if resume is None else 'stop_resume_01.json')
        suffix='' if resume is None else '_resume_01'
        try:
            with (root/('worker'+suffix+'.stdout')).open('xb') as out,(root/('worker'+suffix+'.stderr')).open('xb') as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,start_new_session=True)
                child_id=process_identity(child.pid);write_new(root/('child_identity'+suffix+'.json'),child_id)
                while True:
                    now=time.monotonic();elapsed=carry_seconds+now-attempt_origin['start']['monotonic']
                    if now-last>p['rss_gap_s'] and last>0:reason='RSS_observation_gap'
                    rss=rss_observation(os.getpid(),attempt_origin['supervisor'])
                    system=system_observation() if now-last_system>=p['system_period_s'] else None
                    if system is not None:last_system=now
                    append(resources,dict(run_id=run_id,attempt=attempt,elapsed=elapsed,rss=rss,system=system,peak_kind='sampled maximum; not OS peak'))
                    snap=byte_snapshot(root)
                    for name,size in snap.items():total+=max(0,size-high.get(name,0));high[name]=max(size,high.get(name,0))
                    reason=reason or st.check(elapsed,rss,system,total);last=now
                    if reason:
                        write_new(stopfile,dict(reason=reason,at=clock(),elapsed=elapsed,bytes=total,child=child_id));break
                    cp_exit=child.poll()
                    if cp_exit is not None:break
                    time.sleep(max(0,p['rss_period_s']-(time.monotonic()-now)))
                if reason is not None and child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                cp_exit=child.wait(timeout=5);reap=clock()
                if reason is None and cp_exit!=0:
                    reason='worker_nonzero_exit';write_new(stopfile,dict(reason=reason,at=clock(),exit=cp_exit))
        except BaseException as e:
            reason=reason or ('supervisor_'+type(e).__name__+': '+str(e)[:500])
            try:
                if not stopfile.exists():write_new(stopfile,dict(reason=reason,at=clock()))
            finally:
                if child is not None:
                    try:
                        if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                        cp_exit=child.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);cp_exit=child.wait(timeout=5)
                    reap=clock()
        for name,size in byte_snapshot(root).items():total+=max(0,size-high.get(name,0));high[name]=max(size,high.get(name,0))
        if total>=p['output_bytes'] and reason is None:reason='final_output_byte_limit'
        summary=dict(run_id=run_id,attempt=attempt,status='completed' if reason is None else 'stopped',reason=reason,exit=cp_exit,reap=reap,
            elapsed=carry_seconds+time.monotonic()-attempt_origin['start']['monotonic'],new_bytes=total,carry_seconds=carry_seconds,carry_bytes=carry_bytes,
            sampled_tree_rss_max_bytes=st.max_rss,peak_kind='sampled maximum',resources=reference(resources) if resources.exists() else None,
            parent_receipt=resume['parent_receipt'] if resume else None,original_failure_preserved=parent['reason'] if parent else None,
            output_accounting='cumulative per-file highwater growth plus conservative carry; no decrement on deletion; manual stopped repair time excluded from machine runtime')
        write_new(root/('supervisor_receipt'+suffix+'.json'),summary,True)
        return summary

