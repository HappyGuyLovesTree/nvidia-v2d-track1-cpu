"""Shared deterministic contracts; no model imports or calls here."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def load(p):return json.loads(Path(p).read_text())
def iou(a,b):
 a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
 wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));inter=float(np.prod(wh))
 union=float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter)
 return inter/union if union>0 else 0.
def initial_key(c):return [-c['score'],-c['area'],*c['xyxy'],c['index']]
class Tracker:
 def __init__(self):self.last=None;self.last_frame=None;self.lost=0;self.previous=None;self.source=-1;self.track=-1
 def step(self,fid,candidates,w,h):
  discontinuity=self.previous is not None and fid!=self.previous+1
  if self.previous is None or discontinuity:
   self.last=None;self.last_frame=None;self.lost=0;self.source+=1
  self.previous=fid
  valid=[c for c in candidates if c['valid'] and c['class_id']==0]
  for c in valid:
   c['initial_key']=initial_key(c);c['iou']=iou(c['xyxy'],self.last) if self.last is not None else None
   c['association_key']=[-c['iou'],*initial_key(c)] if self.last is not None else None
  chosen=None;reason=None;branch='full_no_detection';bbox=[0,0,w,h];score=None
  had_state=self.last is not None
  if had_state:
   linked=[c for c in valid if c['iou']>=.30]
   if linked:chosen=min(linked,key=lambda c:c['association_key']);branch='associated_detection'
   else:
    self.lost+=1
    if self.lost<=2:branch='held_bbox';bbox=self.last.copy();reason='no_iou_match'
    else:
     self.last=None;self.last_frame=None;self.lost=0
     if valid:chosen=min(valid,key=initial_key);branch='reinitialized_identity_uncertain'
     else:reason='third_loss_cleared_state'
  else:
   if valid:chosen=min(valid,key=initial_key);branch='initialized_detection'
   else:reason='no_valid_person'
  if chosen is not None:
   if branch in ('initialized_detection','reinitialized_identity_uncertain'):self.track+=1
   self.last=chosen['xyxy'].copy();self.last_frame=fid;self.lost=0;bbox=self.last.copy();score=chosen['score']
  return {'frame_index':fid,'source_segment':self.source,'source_discontinuity':discontinuity,'track_segment':self.track,
   'branch':branch,'bbox':bbox,'score':score,'selected_index':None if chosen is None else chosen['index'],
   'lost_count':self.lost,'last_observed_bbox':self.last,'last_observed_frame':self.last_frame,
   'true_gap':None if self.last_frame is None else fid-self.last_frame,'reason':reason,'candidates':candidates}
def candidate(index,box,score,class_id,w,h,**extra):
 a=np.asarray(box,dtype=np.float64);reasons=[]
 if not np.isfinite(a).all() or not math.isfinite(score):reasons.append('nonfinite')
 else:
  a=np.clip(a,[0,0,0,0],[w,h,w,h])
  if a[2]<=a[0] or a[3]<=a[1]:reasons.append('nonpositive_area')
 return {'index':index,'xyxy':a.tolist() if np.isfinite(a).all() else None,'score':float(score) if math.isfinite(score) else None,
  'class_id':int(class_id),'valid':not reasons,'rejection_reasons':reasons,
  'area':float(np.prod(a[2:]-a[:2])) if not reasons else 0.,**extra}
def nearest(fid,ids):return min(ids,key=lambda x:(abs(x-fid),x))
def proxy_geometry(raw):
 J=raw['joints_3d'][0].astype(np.float64)+raw['cam_trans'][0].astype(np.float64)
 l=J[[62,49,53,57,61]].mean(0);r=J[[41,28,32,36,40]].mean(0);c=(l+r)/2
 d=float(np.linalg.norm(r-l));b=float((np.linalg.norm(J[49]-J[61])+np.linalg.norm(J[28]-J[40]))/2)
 a=((l-J[7])+(r-J[8]))/2;an=float(np.linalg.norm(a))
 x=(r-l)/d if d>0 and np.isfinite(d) else None
 R=None;un=None
 if x is not None:
  u=a-x*np.dot(x,a);un=float(np.linalg.norm(u))
  if np.isfinite(J[[62,49,53,57,61,41,28,32,36,40,7,8]]).all() and min(d,b,an,un)>0 and d/b>.5 and un/an>.1:
   y=u/un;R=np.column_stack([x,y,np.cross(x,y)])
   if np.max(np.abs(R.T@R-np.eye(3)))>1e-6 or abs(np.linalg.det(R)-1)>1e-6:R=None
 return {'J':J,'c':c,'x':x,'R':R,'d':d,'b':b,'a_norm':an,'u_norm':un}
def complete_proxy(geoms):
 goodR=[f for f,g in geoms.items() if g['R'] is not None];goodC=[f for f,g in geoms.items() if np.isfinite(g['c']).all()]
 dims=[f for f,g in geoms.items() if np.isfinite([g['d'],g['b']]).all() and min(g['d'],g['b'])>0]
 if not dims:raise ValueError('no_input_driven_mesh_dimensions')
 D=np.array([np.median([geoms[f]['d'] for f in dims]),np.median([geoms[f]['b'] for f in dims]),np.median([geoms[f]['b'] for f in dims])])
 result={}
 for f,g in geoms.items():
  R=g['R'];c=g['c'];rp=f;cp=f;kind='formula'
  if R is None:
   if goodR:rp=nearest(f,goodR);R=geoms[rp]['R'];kind='nearest_formula_rotation'
   elif g['x'] is not None and np.isfinite(g['x']).all():
    x=g['x'];axis=np.eye(3)[int(np.argmin(np.abs(x)))];y=axis-x*np.dot(axis,x);y/=np.linalg.norm(y);R=np.column_stack([x,y,np.cross(x,y)]);kind='camera_axis_prior';rp=None
   else:R=np.eye(3);kind='identity_orientation_prior';rp=None
  if not np.isfinite(c).all():
   if not goodC:raise ValueError('no_finite_translation')
   cp=nearest(f,goodC);c=geoms[cp]['c']
  assert np.isfinite(R).all() and np.isfinite(c).all() and np.max(np.abs(R.T@R-np.eye(3)))<=1e-6 and abs(np.linalg.det(R)-1)<=1e-6
  result[f]={'R':R,'t':c,'rotation_source':kind,'rotation_parent':rp,'rotation_distance':None if rp is None else abs(rp-f),
   'translation_parent':cp,'translation_distance':abs(cp-f),'held_translation':cp!=f}
 return D,dims,result
def box_mesh(D):
 v=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],float)*D/2
 faces=np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[3,7,6],[3,6,2],[0,4,7],[0,7,3],[1,2,6],[1,6,5]],np.int64)
 return v,faces
