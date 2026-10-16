"""R011 durable transactions and shared budget. No model imports."""
from pathlib import Path
from contextlib import contextmanager
import datetime, fcntl, hashlib, io, json, os, time, uuid
import numpy as np
from common import sha, load
from tracking import digest

TZ=datetime.timezone(datetime.timedelta(hours=8))
def now(): return datetime.datetime.now(TZ).isoformat()
def require(ok, message):
    if not ok: raise ValueError(message)
class BudgetStop(BaseException): pass

def json_bytes(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
def atomic_bytes(path, data, exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with tmp.open('xb') as f: f.write(data);f.flush();os.fsync(f.fileno())
        if exclusive:
            os.link(tmp,path);tmp.unlink()
        else: os.replace(tmp,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if tmp.exists(): tmp.unlink()
def atomic_json(path,value,exclusive=False): atomic_bytes(path,json_bytes(value),exclusive)
def npz_bytes(**values):
    require(all(np.asarray(v).dtype.kind!='O' for v in values.values()),'object arrays forbidden')
    b=io.BytesIO();np.savez_compressed(b,**values);return b.getvalue()
def array_hash(a):
    a=np.ascontiguousarray(a)
    return digest({'shape':list(a.shape),'dtype':str(a.dtype),'bytes_sha256':hashlib.sha256(a.tobytes()).hexdigest()})
def arrays_hash(values): return digest({k:array_hash(v) for k,v in values.items()})
def read_npz(path):
    with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def ref(path): return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def verify_ref(r):
    require(sha(r['path'])==r['sha256'],'artifact hash mismatch: '+r['path']);return Path(r['path'])

def append_line(path, row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);b=json_bytes(row)+b'\n'
    with path.open('ab') as f:
        offset=f.tell();f.write(b);f.flush();os.fsync(f.fileno())
    return {'path':str(path.resolve()),'offset':offset,'length':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read_line_ref(r):
    with Path(r['path']).open('rb') as f:f.seek(r['offset']);b=f.read(r['length'])
    require(hashlib.sha256(b).hexdigest()==r['sha256'],'journal record mismatch');return json.loads(b)

class Owner:
    """Kernel lock is authoritative; immutable attempt records preserve owner history."""
    def __init__(self,root,stage,episode):
        self.root=Path(root);self.stage=stage;self.episode=episode
    def __enter__(self):
        import psutil
        self.root.mkdir(parents=True,exist_ok=True);self.fd=(self.root/'owner.lock').open('a+')
        try:fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:self.fd.close();raise ValueError('active owner refuses another attempt')
        self.attempt=uuid.uuid4().hex;prev=load(self.root/'owner_latest.json') if (self.root/'owner_latest.json').exists() else None
        if prev and not (self.root/'attempts'/f"{prev['attempt']}_end.json").exists():
            try:
                old=psutil.Process(prev['pid'])
                if old.is_running() and old.status()!=psutil.STATUS_ZOMBIE and abs(old.create_time()-prev['pid_started'])<.001:
                    fcntl.flock(self.fd,fcntl.LOCK_UN);self.fd.close();raise ValueError('live prior owner identity refuses recovery')
            except psutil.NoSuchProcess:pass
        self.record={'attempt':self.attempt,'parent_attempt':prev['attempt'] if prev else None,'pid':os.getpid(),'pid_started':psutil.Process().create_time(),'stage':self.stage,'episode':self.episode,'started_at':now()}
        atomic_json(self.root/'attempts'/f'{self.attempt}.json',self.record,True);atomic_json(self.root/'owner_latest.json',self.record)
        return self
    def __exit__(self,typ,val,tb):
        atomic_json(self.root/'attempts'/f'{self.attempt}_end.json',{'attempt':self.attempt,'finished_at':now(),'outcome':'completed' if typ is None else 'interrupted' if issubclass(typ,BudgetStop) else 'failed','error':None if val is None else repr(val)},True)
        fcntl.flock(self.fd,fcntl.LOCK_UN);self.fd.close()

class Ledger:
    def __init__(self,root,cfg):
        self.root=Path(root);self.cfg=cfg;self.directory=self.root/'budget';self.directory.mkdir(parents=True,exist_ok=True)
        self.state_path=self.directory/'state.json';self.journal=self.directory/'calls.jsonl'
    @contextmanager
    def edit(self):
        with (self.directory/'ledger.lock').open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            s=load(self.state_path) if self.state_path.exists() else {'schema':'r11_budget_v1','counts':{},'episode_counts':{},'pending':{},'written_bytes':{},'created_at':now()}
            try:yield s
            finally:atomic_json(self.state_path,s);fcntl.flock(f,fcntl.LOCK_UN)
    def check_time(self,model=False,new_episode=False):
        t=datetime.datetime.now(TZ)
        if model and self.state_path.exists() and load(self.state_path).get('stop_reason'):raise BudgetStop('latched_resource_stop:'+load(self.state_path)['stop_reason'])
        end=datetime.datetime.fromisoformat(self.cfg['deadlines']['hard_stop'])
        if t>=end:raise BudgetStop('absolute_hard_stop')
        if model and t>=datetime.datetime.fromisoformat(self.cfg['deadlines'].get('candidate_stop',self.cfg['deadlines']['candidate_freeze'])):raise BudgetStop('candidate_stop')
        if new_episode and t>=datetime.datetime.fromisoformat(self.cfg['deadlines']['new_episode_stop']):raise BudgetStop('new_episode_stop')
    def snapshot(self):
        with self.edit() as s:
            # A crash can persist call_end before the compact state file. Closure is
            # decided from durable journal evidence, never from raw-file existence.
            if s['pending'] and self.journal.exists():
                with self.journal.open('rb') as f:
                    for b in f:
                        require(b.endswith(b'\n'),'torn call journal: preserve and investigate')
                        row=json.loads(b);cid=row.get('call_id')
                        if row.get('kind')=='call_end' and cid in s['pending']:
                            start=s['pending'][cid]
                            require(all(row[k]==start[k] for k in ('model_kind','episode','frame','attempt','stage','input_identity')),'closure/start identity conflict')
                            del s['pending'][cid];key=start['model_kind']+':'+row['outcome'];s.setdefault('outcomes',{})[key]=s.get('outcomes',{}).get(key,0)+1
            return json.loads(json.dumps(s))
    def reserve_bytes(self,category,n,write_bytes=None):
        self.check_time()
        with self.edit() as s:
            limit=self.cfg['storage_limits'][category]
            used=s['written_bytes'].get(category,0)
            if used+n>limit:
                s['stop_reason']='storage_limit:'+category;raise BudgetStop('storage_limit:'+category)
            observed=s.get('global_observation')
            if observed and n>0 and observed['global_bytes']+max(0,sum(s['written_bytes'].values())-observed['local_accounted_bytes'])+n>self.cfg['global_new_file_limit']-67108864:
                s['stop_reason']='global_file_headroom';raise BudgetStop('global_file_headroom')
            s['written_bytes'][category]=max(0,used+n)
            s.setdefault('cumulative_write_bytes',{})[category]=s.get('cumulative_write_bytes',{}).get(category,0)+(max(0,n) if write_bytes is None else write_bytes)
    def invoke(self,kind,fn,*,episode,frame,attempt,stage,n,output_identity=None,input_identity=None):
        self.check_time(model=True)
        if kind in ('dnn_load','ort_session_load','rig_load'):
            prior=self.snapshot()['episode_counts']
            if not any(prior.get(str(episode)+':'+k,0) for k in ('dnn_load','ort_session_load','rig_load')):self.check_time(model=True,new_episode=True)
        limits=self.cfg['call_limits'];key=str(episode)+':'+kind
        with self.edit() as s:
            if s['counts'].get(kind,0)>=limits[kind]:raise BudgetStop('global_call_limit:'+kind)
            ep_limit=n*2 if kind=='rig' else n if kind in ('detector','hmr') else None
            if ep_limit is not None and s['episode_counts'].get(key,0)>=ep_limit:raise BudgetStop('episode_call_limit:'+key)
            s['counts'][kind]=s['counts'].get(kind,0)+1;s['episode_counts'][key]=s['episode_counts'].get(key,0)+1
            cid=uuid.uuid4().hex;row={'kind':'call_start','call_id':cid,'model_kind':kind,'episode':episode,'frame':frame,'attempt':attempt,'stage':stage,'time':now(),'ordinal':s['counts'][kind],'input_identity':input_identity}
            # Reservation precedes the forward; pending stays charged even if journaling fails.
            s['pending'][cid]=row
            atomic_json(self.state_path,s);append_line(self.journal,row)
        started=time.monotonic()
        try:result=fn()
        except BudgetStop:raise
        except BaseException as e:
            if not isinstance(e,Exception):raise
            self._end(cid,'failed',repr(e),time.monotonic()-started,None)
            raise ClosedCallError(repr(e),cid) from e
        identity=output_identity(result) if output_identity else None
        endref=self._end(cid,'returned',None,time.monotonic()-started,identity)
        return result,endref
    def _end(self,cid,outcome,error,elapsed,identity):
        with self.edit() as s:
            start=s['pending'][cid];row={**start,'kind':'call_end','time':now(),'outcome':outcome,'error':error,'elapsed_s':elapsed,'output_identity':identity}
            r=append_line(self.journal,row);del s['pending'][cid]
            k=start['model_kind']+':'+outcome;s.setdefault('outcomes',{})[k]=s.get('outcomes',{}).get(k,0)+1
        return r
    def failed_end(self,cid):
        # ClosedCallError carries its durable end reference, found without guessing from raw.
        with self.journal.open('rb') as f:
            while True:
                offset=f.tell();b=f.readline()
                if not b:break
                row=json.loads(b)
                if row.get('call_id')==cid and row['kind']=='call_end':return {'path':str(self.journal.resolve()),'offset':offset,'length':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        raise ValueError('closed failure has no journal evidence')
    def unknown_for(self,episode,frame=None):
        return [r for r in self.snapshot()['pending'].values() if r['episode']==episode and (frame is None or r['frame']==frame)]
class ClosedCallError(Exception):
    def __init__(self,message,call_id):super().__init__(message);self.call_id=call_id

class Writer:
    def __init__(self,ledger):self.ledger=ledger
    def bytes(self,path,b,category='small',exclusive=False):
        path=Path(path);require(not (exclusive and path.exists()),'immutable artifact already exists: '+str(path));old=path.stat().st_size if path.exists() else 0
        delta=len(b)-old;self.ledger.reserve_bytes(category,max(0,delta),write_bytes=len(b));atomic_bytes(path,b,exclusive)
        if delta<0:self.ledger.reserve_bytes(category,delta,write_bytes=0)
        return ref(path)
    def json(self,path,v,category='small',exclusive=False):return self.bytes(path,json_bytes(v),category,exclusive)
    def npz(self,path,category='small',**values):return self.bytes(path,npz_bytes(**values),category)

    def remove(self,path,category):
        path=Path(path)
        if path.exists():
            n=path.stat().st_size;path.unlink();self.ledger.reserve_bytes(category,-n,write_bytes=0)

class FrameStore:
    def __init__(self,episode_root,writer):
        self.root=Path(episode_root);self.writer=writer;self.journal=self.root/'frame_commits.json'
    def records(self):
        result={};previous=None
        if not self.journal.exists():return result
        lines=load(self.journal)
        for row in lines:
            require(row['previous_commit']==previous,'frame journal chain mismatch')
            require(row['frame'] not in result,'duplicate frame commit');verify_ref(row['record']);record=load(row['record']['path'])
            require(record['status']=='frame_committed' and record['frame']==row['frame'],'invalid terminal record')
            result[row['frame']]=record;previous=digest(row)
        return result
    def commit(self,record):
        require(record['hmr_outcome'] in ('hmr_valid','hmr_failed'),'unknown cannot be frame committed')
        end=read_line_ref(record['hmr_call_end']);require(end['kind']=='call_end' and end['model_kind']=='hmr','missing closed HMR call')
        path=self.root/'frames'/f"{record['frame']:06d}.json"
        r=self.writer.json(path,record,'A',exclusive=True)
        self.append_commit(record['frame'],r)
    def append_commit(self,fid,r):
        rows=load(self.journal) if self.journal.exists() else []
        previous=digest(rows[-1]) if rows else None
        row={'frame':fid,'record':r,'previous_commit':previous,'committed_at':now()}
        self.writer.json(self.journal,rows+[row],'A')
    def orphan(self,fid):
        p=self.root/'frames'/f'{fid:06d}.json'
        return load(p) if p.exists() else None
