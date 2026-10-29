"""Generic original R011/R012 R0 adapter; paths come only from a caller manifest.

No private proposal/result cache is an input. Relocation maps only file lookup;
original JSON bytes, parent digests, PTS and numerical values are not rewritten.
"""
import json,hashlib,io,collections
from pathlib import Path
from fractions import Fraction
import numpy as np
from normalize import normalize,FRAME_POINTERS,INPUT_POINTERS
from restricted_json import extract,legacy_content_digest
from control import reference,compact_digest,legacy_array_hash
from geometry import need
from d_common import bit_equal
from geometry_d import params204
from planner import require

DENSE=('pose','scales','shape','object_rotation','object_translation','object_scale')
CALL_FIELDS=['/kind','/call_id','/model_kind','/episode','/frame','/attempt','/stage','/ordinal','/input_identity','/outcome','/output_identity']
EXTRA=['/raw','/hmr_outcome','/hmr_call_end','/transform/K','/input_identity/hmr_tensor_hashes']

def parse_obj(raw):
    vertices=[];faces=[]
    for line in raw.decode().splitlines():
        fields=line.split()
        if not fields or fields[0].startswith('#'):continue
        if fields[0]=='v':
            need(len(fields)==4,'OBJ vertex shape');vertices.append([float(x) for x in fields[1:]])
        elif fields[0]=='f':
            need(len(fields)==4 and all('/' not in x for x in fields[1:]),'OBJ triangle/index contract')
            faces.append([int(x)-1 for x in fields[1:]])
        else:need(False,'unsupported OBJ record')
    v=np.asarray(vertices,np.float64);f=np.asarray(faces,np.int64)
    need(v.ndim==2 and v.shape[1]==3 and len(v)>0 and np.isfinite(v).all(),'OBJ vertex finite/domain')
    need(f.ndim==2 and f.shape[1]==3 and len(f)>0 and np.all(f>=0) and np.all(f<len(v)),'OBJ face domain')
    tri=v[f];need(np.all(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)>0),'OBJ degenerate/padded face')
    return v,f

class Dataset:
    def __init__(self,path):
        self.path=Path(path).resolve();self.base=self.path.parent
        self.config=json.loads(self.path.read_text());need(self.config['schema']=='t1_r0_dataset_v1','dataset schema')
        self.maps=[(a,self.local(b)) for a,b in self.config.get('prefix_map',[])]
        self.entries={};self.ep={};self.records={};self.normalized=[];self.bplans={};self.dense={};self.context={};self.calls={};self.B={}
        need(len({e['episode'] for e in self.config['episodes']})==len(self.config['episodes']) and bool(self.config['episodes']),'dataset unique nonempty episodes')
    def local(self,path):
        p=Path(path);return p.resolve() if p.is_absolute() else (self.base/p).resolve()
    def resolve(self,path):
        text=str(path)
        for old,new in sorted(self.maps,key=lambda x:-len(x[0])):
            if text==old or text.startswith(old.rstrip('/')+'/'):return new/Path(text).relative_to(old)
        return self.local(path)
    def pin(self,path,fields,expected=None):
        p=self.resolve(path);r=reference(p)
        if expected:
            need(r['sha256']==expected['sha256'] and ('bytes' not in expected or r['bytes']==expected['bytes']),'immutable R0 file hash/bytes conflict')
        if str(p) not in self.entries:self.entries[str(p)]=dict(r,fields=list(fields))
        else:self.entries[str(p)]['fields']=sorted(set(self.entries[str(p)]['fields'])|set(fields))
        return r
    def selected(self,path,fields,expected=None):
        r=self.pin(path,fields,expected);b=Path(r['path']).read_bytes()
        need(hashlib.sha256(b).hexdigest()==r['sha256'],'R0 metadata changed during read')
        return extract(b.decode(),fields),r
    def arrays(self,path,fields,expected=None):
        r=self.pin(path,fields,expected);b=Path(r['path']).read_bytes()
        need(hashlib.sha256(b).hexdigest()==r['sha256'],'R0 arrays changed during read')
        with np.load(io.BytesIO(b),allow_pickle=False) as z:
            need(set(fields)<=set(z.files),'R0 NPZ missing field')
            return {k:z[k] for k in fields}
    def controls(self):
        for spec in sorted(self.config['episodes'],key=lambda x:x['episode']):
            e=spec['episode'];run=self.local(spec['run_root']);input_path=run/'inputs'/f'episode_{e:06d}.json'
            ep,inp=self.selected(input_path,INPUT_POINTERS)
            need(ep['schema']=='r11_episode_input_v1' and ep['episode']==e,'R0 episode input')
            ids=ep['required'];runs=[]
            need(ids and ids==sorted(set(ids)) and all(type(f)is int and f>=0 for f in ids),'R0 required frames')
            for f in ids:
                if not runs or f!=runs[-1][-1]+1:runs.append([])
                runs[-1].append(f)
            need(ep['sources']==runs and ep['source_map']=={str(f):i for i,s in enumerate(runs) for f in s},'R0 source partition')
            self.ep[e]=dict(spec=spec,control=ep,run=run)
            journal_path=run/'episode_runs'/f'episode_{e:06d}'/'frame_commits.json'
            self.pin(journal_path,['frame','record','previous_commit','committed_at']);journal=json.loads(journal_path.read_text())
            need([x['frame'] for x in journal]==ids,'R0 exact committed required frames')
            previous_commit=None;previous_frame=None;input_digest=legacy_content_digest(input_path)
            for item in journal:
                need(item['previous_commit']==previous_commit,'R0 commit journal chain')
                previous_commit=compact_digest(item);f=item['frame']
                path=self.resolve(item['record']['path']);expected_path=run/'episode_runs'/f'episode_{e:06d}'/'frames'/f'{f:06d}.json'
                need(path==expected_path,'R0 original frame path')
                record,ref=self.selected(path,FRAME_POINTERS+EXTRA,item['record'])
                original_digest=legacy_content_digest(path)
                entry=dict(ref,episode=e,frame=f,source=ep['source_map'][str(f)],input_ref={'legacy_content_sha256':input_digest},
                    required_source_ref=inp)
                row=normalize(entry,ep,record,previous_frame);previous_frame=original_digest
                self.records[e,f]=dict(record=record,ref=ref,content_digest=original_digest,row=row)
                self.normalized.append(row)
        return self.normalized
    def bplan(self,e):
        if e not in self.bplans:
            er=self.ep[e]['run']/'episode_runs'/f'episode_{e:06d}'
            latest,_=self.selected(er/'B_plan_latest.json',['/path','/sha256','/plan_hash'])
            path=self.resolve(latest['path']);need(path==er/'B'/latest['plan_hash']/'plan.json','R0 B exact plan path')
            self.pin(path,['lexical content hash'],latest)
            need(legacy_content_digest(path)==latest['plan_hash'],'R0 B lexical plan hash')
            self.bplans[e]=(path,latest['plan_hash'])
        return self.bplans[e]
    def slice(self,ref):
        p=self.resolve(ref['path']);need(p.resolve()==p and 0<ref['length']<=16384 and ref['offset']>=0,'original call slice bounds')
        with p.open('rb') as f:f.seek(ref['offset']);raw=f.read(ref['length'])
        need(raw.endswith(b'\n') and hashlib.sha256(raw).hexdigest()==ref['sha256'],'original call slice hash')
        return extract(raw.decode(),CALL_FIELDS)
    def contexts(self,segments):
        endrefs={};logs=collections.defaultdict(set)
        for s in segments:
            for f in s['frames']:
                e=s['key'][0];item=self.records[e,f];r=item['record'];bp,ph=self.bplan(e)
                bplan,_=self.selected(bp,['/parents/'+str(f),'/frames/'+str(f)])
                need(bplan['frames'][str(f)]==item['content_digest'],'R0 B/A frame binding')
                brec,bref=self.selected(bp.parent/'frames'/f'{f:06d}.json',['/frame','/source','/plan_hash','/hmr_fill','/output','/call_ends'])
                need(brec['frame']==f and brec['source']==r['source'] and brec['plan_hash']==ph and bplan['parents'][str(f)]==brec['hmr_fill'],'R0 B parent/source conflict')
                current=brec['hmr_fill']['frame']==f
                row=dict(item['row'],segment_id=s['segment_id'],current=current,parent_closed=bool(r.get('hmr_call_end') and brec.get('call_ends')))
                self.context[e,f]=row;self.B[e,f]=dict(receipt=brec,ref=bref,plan=bp,plan_hash=ph)
                need(row['parent_closed'],'missing original call refs')
                need(len(brec['call_ends'])==2,'original B pair refs')
                refs=[(r['hmr_call_end'],'hmr','A')]+[(x,'rig','B:'+role) for role,x in zip(('baseline','adapted'),brec['call_ends'])]
                for ref,model,stage in refs:
                    need(self.resolve(ref['path'])==self.ep[e]['run']/'budget/calls.jsonl','original call log location')
                    end=self.slice(ref);cid=end['call_id']
                    need(end['kind']=='call_end' and end['model_kind']==model and end['stage']==stage and end['episode']==e and end['frame']==f,'original end identity')
                    if current or model=='rig':need(end['outcome']=='returned','original parent not returned')
                    if model=='hmr':need(end['input_identity']==compact_digest(r['input_identity']['hmr_tensor_hashes']),'original HMR input hash')
                    need(cid not in endrefs,'duplicate selected original call ID');endrefs[cid]=(ref,end)
                    logs[str(self.resolve(ref['path']))].add(cid)
                if current:
                    need(r['hmr_outcome']=='hmr_valid' and r['raw']['sha256']==brec['hmr_fill']['raw_sha256'],'current R0 raw binding')
                    cam=self.arrays(r['raw']['path'],['cam_trans'],r['raw'])['cam_trans']
                    need(cam.shape==(1,3) and cam.dtype==np.float32 and np.isfinite(cam).all(),'R0 cam schema')
                    row['cam']=cam[0].copy()
        starts={};seen_ends=set()
        for path,cids in logs.items():
            with Path(path).open('rb') as f:
                while True:
                    offset=f.tell();raw=f.readline()
                    if not raw:break
                    need(len(raw)<=16384 and raw.endswith(b'\n'),'original call index torn/oversize')
                    cid=extract(raw.decode(),['/call_id']).get('call_id')
                    if cid not in cids:continue
                    value=extract(raw.decode(),CALL_FIELDS)
                    if value['kind']=='call_start':
                        need(cid not in starts,'duplicate original start');starts[cid]=value
                    elif value['kind']=='call_end':
                        ref,end=endrefs[cid];need(cid not in seen_ends and offset==ref['offset'] and len(raw)==ref['length'] and value==end,'original selected end conflict');seen_ends.add(cid)
        need(set(starts)==set(endrefs)==seen_ends,'unclosed original call')
        for cid,start in starts.items():
            ref,end=endrefs[cid]
            for key in ('model_kind','episode','frame','attempt','stage','ordinal','input_identity'):need(start[key]==end[key],'original call start/end parent conflict')
            self.calls[cid]=end
        return self.context
    def load_episode(self,e):
        spec=self.ep[e]['spec']
        if e not in self.dense:
            arrays=self.arrays(spec['dense'],DENSE);mask=self.arrays(spec['mask'],['mask','original_frames'])
            need(mask['original_frames'].tolist()==self.ep[e]['control']['required'],'R0 original mask frame IDs')
            check_dense(arrays,self.ep[e]['control']['required'],mask['mask'])
            objref=self.pin(spec['obj'],['bytes']);v,faces=parse_obj(Path(objref['path']).read_bytes())
            mesh=self.arrays(spec['mesh'],['vertices','faces'])
            need(np.array_equal(v,mesh['vertices']) and np.array_equal(faces,mesh['faces']),'OBJ/source nonpadding mesh mismatch')
            self.dense[e]=(arrays,mask,mesh)
        return self.dense[e]
    def geometry(self,e,f):
        need((e,f) in self.context and self.context[e,f]['current'],'geometry outside current P scope')
        item=self.records[e,f];record=item['record'];b=self.B[e,f];spec=self.ep[e]['spec']
        arrays,mask,mesh=self.load_episode(e)
        need(mask['mask'].dtype==np.bool_ and bool(mask['mask'][f]),'selected R0 required mask')
        old=dict(pose=arrays['pose'][f].copy(),shape=arrays['shape'].copy(),scales=arrays['scales'].copy(),object_rotation=arrays['object_rotation'][f].copy(),
                 object_translation=arrays['object_translation'][f].copy(),object_scale=arrays['object_scale'].copy(),mask=np.asarray(mask['mask'][f]).copy())
        need(all(np.isfinite(v).all() for v in old.values()),'selected R0 finite')
        p,shape=params204(old);bparams=self.arrays(b['receipt']['output']['path'],['adapted_params','R','t'],b['receipt']['output'])
        need(bit_equal(p,bparams['adapted_params']) and np.array_equal(old['object_rotation'],bparams['R']) and np.array_equal(old['object_translation'],bparams['t']),'R0 dense/B exact numeric conflict')
        identity,_=self.selected(b['plan'],['/shape','/scales'])
        need(np.array_equal(shape,np.asarray(identity['shape'],np.float32)) and np.array_equal(old['scales'],np.asarray(identity['scales'],np.float32)),'R0 static identity parent')
        end=self.slice(b['receipt']['call_ends'][1]);expected=compact_digest(dict(params=legacy_array_hash(p),shape=legacy_array_hash(shape[None]),plan=b['plan_hash']))
        need(end['input_identity']==expected,'R0 adapted rig actual input parent')
        return old,self.context[e,f]['cam'].copy(),np.asarray(record['transform']['K'],np.float64),mesh['vertices'].copy(),mesh['faces'].copy()


def check_dense(data,ids,mask):
    require(ids==sorted(set(ids)) and len(ids)>0,'invalid dense frame ids')
    require(data['pose'].ndim==2 and data['pose'].shape[1:]==(136,),'pose shape')
    require(data['object_rotation'].ndim==3 and data['object_rotation'].shape[1:]==(3,3),'rotation shape')
    require(data['object_translation'].ndim==2 and data['object_translation'].shape[1:]==(3,),'translation shape')
    for k in ('pose','object_rotation','object_translation'):
        require(len(data[k])>max(ids) and np.isfinite(data[k][ids]).all(),k+' independent T/finite check')
        require(len(data[k])==len(mask) and np.isnan(data[k][~mask]).all(),k+' dense/mask mismatch')
    require(np.flatnonzero(mask).tolist()==ids,'completed mask not exact')
    require(data['scales'].shape==(68,) and data['shape'].shape==(45,) and np.isfinite(data['scales']).all() and np.isfinite(data['shape']).all(),'identity shape/finite')
    require(np.shape(data['object_scale'])==() and np.isfinite(data['object_scale']) and data['object_scale']>0,'positive scalar scale')

