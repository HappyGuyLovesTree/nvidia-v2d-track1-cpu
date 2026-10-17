"""Original portable IO and bounded subprocess helpers; Apache-2.0."""
from pathlib import Path
import datetime, hashlib, json, math, os, subprocess, sys, time, traceback, uuid

class ContractError(ValueError):
    pass

def require(value, message):
    if not value:
        raise ContractError(message)

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def ref(path):
    p=Path(path).resolve()
    return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}

def load(path):
    return json.loads(Path(path).read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def save_new(path, value):
    """Exclusive complete evidence file, no overwrites of an existing receipt."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    b=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:
        f.write(b);f.flush();os.fsync(f.fileno())
    return ref(path)

class Paths:
    def __init__(self, workspace, overrides=None):
        self.workspace=Path(workspace).expanduser().resolve()
        self.overrides=overrides or {}
        require(all(isinstance(k,str) and isinstance(v,str) for k,v in self.overrides.items()),'path overrides must be string mappings')

    def output(self, logical):
        p=Path(logical)
        require(not p.is_absolute() and '..' not in p.parts and str(p) not in ('','.'),'output needs a relative logical file/directory')
        result=(self.workspace/p).resolve()
        require(result.is_relative_to(self.workspace),'output escapes workspace through a symlink')
        return result

    def asset(self, key, logical):
        return Path(self.overrides[key]).expanduser().resolve() if key in self.overrides else self.output(logical)

def read_paths(path, workspace=None):
    config=load(path)
    require(config.get('schema')=='r12_public_paths_v1','unsupported paths schema')
    root=workspace or config.get('workspace')
    require(root,'pass --workspace or set workspace in local paths configuration')
    return Paths(root,config.get('asset_overrides',{})),config

def run_logged(argv, directory, *, timeout, max_output_bytes=16*1024*1024, env=None, cwd=None, cleanup_timeout=2.0):
    """Bound stdout/stderr files, kill/reap the directly started process on failure.

    ffprobe and the stock packer are each direct children. This does not replace
    C3's inference process-tree supervisor or claim descendant-tree cleanup.
    """
    require(math.isfinite(timeout) and timeout>0 and math.isfinite(cleanup_timeout) and cleanup_timeout>0
            and type(max_output_bytes) is int and max_output_bytes>0,'finite positive subprocess/cleanup limits required')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();p=None;returncode=None;of=ef=None;reaped=False
    primary=None;cleanup_poll_error=None;kill_error=None;wait_errors=[];receipt_errors=[]
    cleanup={'budget_s':cleanup_timeout,'status':'not_spawned','kill_attempted':False,'wait_attempted':False,'reaped':False}
    out=directory/'stdout';err=directory/'stderr'
    def failure(exc,operation):
        return {'operation':operation,'type':type(exc).__name__,'errno':getattr(exc,'errno',None),'message':str(exc),
                'traceback':''.join(traceback.format_exception(type(exc),exc,exc.__traceback__))}
    def first_error():
        return primary or cleanup_poll_error or kill_error or next(iter(wait_errors),None) or next(iter(receipt_errors),None)
    try:
        try:
            save_new(directory/'started.json',{'started_at':now(),'argv':[str(a) for a in argv],'cwd':str(cwd) if cwd else None,
                     'timeout_s':timeout,'max_output_bytes':max_output_bytes,'cleanup_timeout_s':cleanup_timeout})
        except BaseException as exc:
            receipt_errors.append(failure(exc,'started_receipt_write'));raise
        # Explicit closes below cannot replace a timeout/spawn/process error.
        of=out.open('xb');ef=err.open('xb')
        p=subprocess.Popen([str(a) for a in argv],stdout=of,stderr=ef,env=env,cwd=cwd)
        while p.poll() is None:
            if time.monotonic()-started>=timeout:
                raise TimeoutError('direct child exceeded wall limit')
            if out.stat().st_size+err.stat().st_size>max_output_bytes:
                raise ContractError('direct child exceeded output byte limit')
            time.sleep(.02)
        try:
            # poll already observed exit; do not introduce an unbounded normal wait.
            returncode=p.wait(timeout=0);reaped=True
        except BaseException as exc:
            wait_errors.append(failure(exc,'normal_wait'));raise
        require(out.stat().st_size+err.stat().st_size<=max_output_bytes,'direct child exceeded output byte limit')
        require(returncode==0,'direct child returned nonzero')
    except BaseException as exc:
        primary=failure(exc,'primary')
    finally:
        if p is not None and not reaped:
            cleanup_start=time.monotonic();cleanup_deadline=cleanup_start+cleanup_timeout
            cleanup.update(status='unconfirmed',poll_result='unknown')
            try:
                observed=p.poll();cleanup['poll_result']='running' if observed is None else 'exited'
                cleanup['observed_returncode']=observed
            except BaseException as exc:
                cleanup_poll_error=failure(exc,'cleanup_poll')
            # Signal only this returned child handle after a confirmed running poll.
            # Unknown poll -> no signal. Never find a PID/name or signal a group.
            if cleanup['poll_result']=='running':
                cleanup['kill_attempted']=True
                try:p.kill()
                except BaseException as exc:kill_error=failure(exc,'cleanup_kill')
            remaining=cleanup_deadline-time.monotonic()
            if remaining>0:
                cleanup['wait_attempted']=True;cleanup['wait_timeout_s']=remaining
                try:returncode=p.wait(timeout=remaining);reaped=True
                except BaseException as exc:wait_errors.append(failure(exc,'cleanup_wait'))
            else:
                wait_errors.append(failure(TimeoutError('cleanup allowance exhausted before wait'),'cleanup_wait_budget'))
            cleanup.update(status='reaped' if reaped else 'unconfirmed',reaped=reaped,elapsed_s=time.monotonic()-cleanup_start)
        elif reaped:
            cleanup.update(status='already_reaped',reaped=True,elapsed_s=0.)
        for label,handle in (('stdout',of),('stderr',ef)):
            if handle is not None:
                try:handle.close()
                except BaseException as exc:receipt_errors.append(failure(exc,label+'_close'))
        refs={}
        for label,path in (('stdout',out),('stderr',err)):
            try:
                if not path.exists():refs[label]=None
                elif p is not None and not reaped:
                    # An unreaped child may still append. Do not hash a moving,
                    # potentially growing log or claim this is its final content.
                    refs[label]={'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':None,'state':'unconfirmed_child_may_write'}
                else:refs[label]=ref(path)
            except BaseException as exc:
                refs[label]=None;receipt_errors.append(failure(exc,label+'_reference'))
        result={'status':'failed' if first_error() else 'completed','finished_at':now(),'elapsed_s':time.monotonic()-started,
                'returncode':returncode,'error':first_error(),'primary_error':primary,'cleanup_poll_error':cleanup_poll_error,
                'kill_error':kill_error,'wait_errors':wait_errors,'receipt_errors':receipt_errors,'cleanup':cleanup,
                'stdout':refs['stdout'],'stderr':refs['stderr'],'log_snapshot_state':'after_reap' if reaped else 'unconfirmed_or_not_spawned',
                'argv':[str(a) for a in argv],'terminal_receipt':{'status':'written','path':str(directory/'finished.json')}}
        try:
            save_new(directory/'finished.json',result)
        except BaseException as exc:
            receipt_errors.append(failure(exc,'finished_receipt_write'))
            fallback=directory/('finished_fallback_'+uuid.uuid4().hex+'.json')
            result.update(status='failed',error=first_error(),terminal_receipt={'status':'fallback_written','path':str(fallback)})
            try:save_new(fallback,result)
            except BaseException as exc:
                receipt_errors.append(failure(exc,'fallback_receipt_write'))
                result['terminal_receipt']={'status':'unavailable','path':None}
                # Disk/permission failure cannot be made durable here. Return the
                # complete errors and attempt stderr without pretending a file exists.
                try:sys.stderr.write(json.dumps(result,ensure_ascii=False)+'\n');sys.stderr.flush()
                except BaseException as exc:receipt_errors.append(failure(exc,'stderr_fallback'))
    return result

def token():
    return datetime.datetime.now().strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:12]
