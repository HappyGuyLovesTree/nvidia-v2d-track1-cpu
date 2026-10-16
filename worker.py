"""One model-budgeted stage per process. Public I/O only; no warmup or retry."""
from __future__ import annotations
import argparse,collections,datetime,hashlib,importlib,importlib.metadata as metadata,math,os,resource,signal,sys,time,traceback
from pathlib import Path
import numpy as np
from common import sha,dump,load,Tracker,candidate,nearest,proxy_geometry,complete_proxy,box_mesh

P=Path(__file__).resolve().parent
EXPECTED={'mhr_params':[1,204],'shape_params':[1,45],'cam_trans':[1,3],'joints_2d':[1,70,2],'joints_3d':[1,70,3]}
SCORED=np.array([1,2,18,35,3,19,36,4,20,37,8,24,110,74,38,113,75,39,76,40,78,42])
F=np.array([1.,-1.,-1.])
COUNTS=collections.Counter();TIMES=collections.Counter();READS=set();START=time.perf_counter()

def event(kind,**kw):
 row={'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'elapsed_s':time.perf_counter()-START,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'kind':kind,**kw}
 with (O/'logs'/f'{STAGE}_events_20261016.jsonl').open('a') as f:f.write(__import__('json').dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def call(name,fn,limit):
 assert COUNTS[name+'_attempted']<limit
 COUNTS[name+'_attempted']+=1;event('call_start',call=name,attempt=COUNTS[name+'_attempted'])
 t=time.perf_counter()
 try:
  result=fn();COUNTS[name+'_completed']+=1;return result
 finally:TIMES[name]+=time.perf_counter()-t;event('call_end',call=name,counts=dict(COUNTS),elapsed_call_s=time.perf_counter()-t)
def runtime(names):
 out={'python':sys.version,'executable':sys.executable,'actual_parallel_workers':'not_traced'}
 for n in names:
  a=importlib.import_module(n);out[n]={'version':getattr(a,'__version__',None),'file':getattr(a,'__file__',None)}
 out['metadata']={}
 for n in ['opencv-python','opencv-python-headless','numpy','torch','trimesh','fast-simplification']:
  try:out['metadata'][n]=metadata.version(n)
  except metadata.PackageNotFoundError:pass
 dump(O/'logs'/f'{STAGE}_runtime_20261016.json',out)
def asset(name):
 a=MAN['assets'][name];assert sha(a['path'])==a['sha256'],name+' changed';READS.add(a['path']);return a['path']
def frame_ids():return list(range(MAN['start'],MAN['stop']+1))
def schema_matches(nodes,expected):
 return {n.name for n in nodes}==set(expected) and all(len(n.shape)==len(expected[n.name]) and (isinstance(n.shape[0],str) or n.shape[0]==1) and n.shape[1:]==expected[n.name][1:] and n.type=='tensor(float)' for n in nodes)
def stageA():
 import cv2,av,onnxruntime as ort
 from vendor.nanodet import NanoDet
 from vendor.letterbox import letterbox,unletterbox
 from vendor.preprocess import _preprocess
 runtime(['numpy','cv2','av','onnxruntime']);cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
 event('threads',cv_requested=1,cv_reported=cv2.getNumThreads(),opencl=cv2.ocl.useOpenCL(),actual_parallel_workers='not_traced')
 so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1;so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
 sess=call('session_load',lambda:ort.InferenceSession(asset('hmr'),sess_options=so,providers=['CPUExecutionProvider']),1)
 dump(O/'logs/model_schema_20261016.json',{'inputs':[{'name':x.name,'shape':x.shape,'type':x.type} for x in sess.get_inputs()], 'outputs':[{'name':x.name,'shape':x.shape,'type':x.type} for x in sess.get_outputs()],'providers':sess.get_providers(),'actual_input_batch':1})
 assert schema_matches(sess.get_inputs(),{'image':[1,3,224,224],'cliff_cond':[1,3]}) and schema_matches(sess.get_outputs(),EXPECTED)
 assert sess.get_providers()==['CPUExecutionProvider']
 detector=call('dnn_load',lambda:NanoDet(asset('nanodet'),.35,.6,cv2.dnn.DNN_BACKEND_OPENCV,cv2.dnn.DNN_TARGET_CPU),1)
 dump(O/'logs/dnn_contract_20261016.json',{'output_names':list(detector.net.getUnconnectedOutLayersNames()),'mean':detector.mean.reshape(-1).tolist(),'std':detector.std.reshape(-1).tolist(),'strides':list(detector.strides),'score':.35,'nms':.6,'backend':'OPENCV','target':'CPU','all_class_argmax_class_agnostic_nms':True})
 container=av.open(asset('video'));stream=container.streams.video[0];stream.codec_context.thread_count=1;stream.codec_context.thread_type='NONE'
 dump(O/'source/stream_20261016.json',{'index':stream.index,'time_base':str(stream.time_base),'average_rate':str(stream.average_rate),'frames_declared':stream.frames,'decoder_threads':stream.codec_context.thread_count,'decoder_thread_type':str(stream.codec_context.thread_type)})
 tracker=Tracker();records=[];prev_time=None;shape=None;tdecode=time.perf_counter()
 for fid,frame in enumerate(container.decode(stream)):
  if fid>MAN['stop']:break
  assert frame.pts is not None
  pt=frame.pts*frame.time_base
  assert prev_time is None or pt>prev_time,'nonincreasing PTS'
  prev_time=pt
  if fid<MAN['start']:continue
  rgb=np.ascontiguousarray(cv2.cvtColor(frame.to_ndarray(format='bgr24'),cv2.COLOR_BGR2RGB));h,w=rgb.shape[:2]
  if shape is None:shape=rgb.shape
  assert rgb.shape==shape and rgb.dtype==np.uint8
  identity={'frame_index':fid,'stream_index':stream.index,'pts':frame.pts,'time_base':str(frame.time_base),'seconds':float(pt),'height':h,'width':w,'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'video_sha256':MAN['assets']['video']['sha256']}
  dump(O/'source'/f'frame_{fid:06d}_identity.json',identity)
  assert cv2.imwrite(str(O/'source'/f'frame_{fid:06d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
  img,scale=letterbox(rgb);blob=detector.pre_process(img)
  assert blob.shape==(1,3,416,416) and blob.dtype==np.float32 and np.isfinite(blob).all()
  np.savez_compressed(O/'detector'/f'input_{fid:06d}.npz',image=blob,letterbox_scale=scale)
  def detect():
   detector.net.setInput(blob);return detector.net.forward(detector.net.getUnconnectedOutLayersNames())
  detraw=call('detector',detect,32)
  np.savez_compressed(O/'detector'/f'raw_{fid:06d}.npz',**{f'output_{i}':v for i,v in enumerate(detraw)})
  preds=detector.post_process(detraw)
  if preds.size==0:preds=np.empty((0,6))
  assert preds.ndim==2 and preds.shape[1]==6
  cs=[];top,left,newh,neww=scale
  for index,row in enumerate(preds):
   box=row[:4];valid=np.isfinite(row).all()
   inv=[float((box[0]-left)*w/neww),float((box[1]-top)*h/newh),float((box[2]-left)*w/neww),float((box[3]-top)*h/newh)] if valid else None
   xy=unletterbox(box,(h,w),scale) if valid else [float('nan')]*4
   c=candidate(index,xy,float(row[4]),int(row[5]) if valid else -1,w,h,letterbox_xyxy=box.tolist() if valid else None,inverse_float_unclipped=inv,paired_inverse_int32=np.asarray(xy,dtype=np.int32).tolist() if valid else None)
   cs.append(c)
  rec=tracker.step(fid,cs,w,h);rec['identity']=identity;rec['letterbox_scale']=scale
  dump(O/'detector'/f'decision_{fid:06d}.json',rec)
  crop,sx,sy,ss,cliff=_preprocess(rgb,np.asarray(rec['bbox'],np.float32),h,w)
  assert crop.shape==(3,224,224) and crop.dtype==np.float32 and cliff.shape==(3,) and cliff.dtype==np.float32 and np.isfinite(crop).all() and np.isfinite(cliff).all()
  np.savez_compressed(O/'hmr'/f'input_{fid:06d}.npz',image=crop[None],cliff_cond=cliff[None])
  # Reconstruct the actual normalised crop for inspection, without a second model call.
  crop_rgb=np.clip((crop.transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406]))*255,0,255).astype(np.uint8)
  assert cv2.imwrite(str(O/'hmr'/f'crop_{fid:06d}.png'),cv2.cvtColor(crop_rgb,cv2.COLOR_RGB2BGR))
  ix1,iy1,ix2,iy2=math.floor(sx),math.floor(sy),math.ceil(sx+ss),math.ceil(sy+ss)
  K=np.array([[math.hypot(w,h),0,w/2],[0,math.hypot(w,h),h/2],[0,0,1.]])
  transform={'frame_index':fid,'bbox':rec['bbox'],'square_float':[sx,sy,ss],'integer_extent':[ix1,iy1,ix2,iy2],
   'padding_ltrb':[max(0,-ix1),max(0,-iy1),max(0,ix2-w),max(0,iy2-h)],'source_extent':[max(0,ix1),max(0,iy1),min(w,ix2),min(h,iy2)],
   'wrapper_crop_to_original':[[ss/224,0,sx],[0,ss/224,sy],[0,0,1]],'integer_resize_to_original':[[(ix2-ix1)/224,0,ix1],[0,(iy2-iy1)/224,iy1],[0,0,1]],'cliff':cliff.tolist(),'K':K.tolist(),'K_source':'virtual_hypot_not_measured'}
  dump(O/'hmr'/f'transform_{fid:06d}.json',transform)
  try:
   vals=call('hmr',lambda:sess.run(list(EXPECTED),{'image':crop[None],'cliff_cond':cliff[None]}),32)
  except Exception as e:
   rec['hmr_error']=repr(e);rec['raw_valid']=False;event('hmr_frame_failure',frame_index=fid,error=repr(e))
  else:
   raw=dict(zip(EXPECTED,vals));np.savez_compressed(O/'hmr'/f'raw_{fid:06d}.npz',**raw)
   checks={k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(np.isfinite(v).all())} for k,v in raw.items()}
   assert all(list(raw[k].shape)==EXPECTED[k] and raw[k].dtype==np.float32 for k in raw),'deterministic output schema error'
   rec['checks']=checks;rec['raw_valid']=all(c['finite'] for c in checks.values());rec['raw_sha256']=sha(O/'hmr'/f'raw_{fid:06d}.npz')
  records.append(rec);dump(O/'source/frames_20261016.json',records)
  event('frame_completed',frame_index=fid,branch=rec['branch'],raw_valid=rec['raw_valid'])
 container.close();TIMES['decode_and_loop_wall']=time.perf_counter()-tdecode
 assert [r['frame_index'] for r in records]==frame_ids() and COUNTS['detector_attempted']==len(frame_ids()) and COUNTS['hmr_attempted']==len(frame_ids())
 dump(O/'source/final_tracker_state_20261016.json',{'state':tracker.__dict__,'config_sha256':MAN['config_sha256'],'commit':MAN['commit'],'video_sha256':MAN['assets']['video']['sha256'],'last_raw_sha256':records[-1].get('raw_sha256')})
 return {'frames':len(records),'raw_valid':sum(r['raw_valid'] for r in records),'branches':dict(collections.Counter(r['branch'] for r in records))}

def stageB():
 import torch
 runtime(['numpy','torch']);torch.set_num_threads(1);torch.set_num_interop_threads(1)
 event('threads',torch_intra=torch.get_num_threads(),torch_inter=torch.get_num_interop_threads())
 ids=frame_ids();records=load(O/'source/frames_20261016.json');raw={}
 for f in ids:
  p=O/'hmr'/f'raw_{f:06d}.npz'
  if p.exists():raw[f]=dict(np.load(p))
 S=[f for f,a in raw.items() if all(a[k].shape==tuple(EXPECTED[k]) and np.isfinite(a[k]).all() for k in ('mhr_params','shape_params','cam_trans'))]
 if not S:raise ValueError('no_identity_estimate')
 scales=np.median(np.stack([raw[f]['mhr_params'][0,136:] for f in S]),axis=0).astype(np.float32)
 shape=np.median(np.stack([raw[f]['shape_params'][0] for f in S]),axis=0).astype(np.float32)
 np.savez_compressed(O/'identity/fixed_identity_20261016.npz',scales=scales,shape=shape,source_frames=S)
 dump(O/'identity/manifest_20261016.json',{'rule':'componentwise_numpy_median_finite_204_45_cam','frames':S,'sample_count':len(S),'identity_sha256':sha(O/'identity/fixed_identity_20261016.npz'),'scales':scales.tolist(),'shape':shape.tolist(),'parents':{str(f):sha(O/'hmr'/f'raw_{f:06d}.npz') for f in S}})
 valid=[f for f,a in raw.items() if all(a[k].shape==tuple(EXPECTED[k]) and np.isfinite(a[k]).all() for k in EXPECTED)]
 if not valid:raise ValueError('no_finite_five_outputs_in_source_segment')
 # Current authorised interval is a single decoded source segment. Never cross a gap.
 assert len({r['source_segment'] for r in records})==1 and ids==list(range(ids[0],ids[-1]+1))
 parents={f:f if f in valid else nearest(f,valid) for f in ids}
 dump(O/'identity/hmr_fill_20261016.json',{'records':[{'frame_index':f,'parent_frame':parents[f],'distance':abs(f-parents[f]),'filled':f!=parents[f],'parent_sha256':sha(O/'hmr'/f'raw_{parents[f]:06d}.npz')} for f in ids]})
 effective={f:raw[parents[f]] for f in ids};geoms={f:proxy_geometry(a) for f,a in effective.items()};D,dimids,proxies=complete_proxy(geoms);verts,faces=box_mesh(D)
 np.savez_compressed(O/'proxy/canonical_mesh_20261016.npz',vertices=verts,faces=faces,dimensions=D)
 dump(O/'proxy/mesh_manifest_20261016.json',{'dimensions_metres':D.tolist(),'dimension_frames':dimids,'d_values':[geoms[f]['d'] for f in dimids],'b_values':[geoms[f]['b'] for f in dimids],'sample_count':len(dimids),'dimension_rule':'median_d_median_b_median_b','origin':'box_center','units':'metres','basis':'right_handed_local','square_cross_section_prior':True,'mesh_sha256':sha(O/'proxy/canonical_mesh_20261016.npz'),'parents':{str(f):sha(O/'hmr'/f'raw_{parents[f]:06d}.npz') for f in ids}})
 geomrows=[]
 for f,g in geoms.items():
  pr=proxies[f];R,t=pr['R'],pr['t'];camera=verts@R.T+t;other=(R@verts.T).T+t
  residual=float(np.max(np.abs(camera-other)));assert residual<=1e-10
  np.savez_compressed(O/'proxy'/f'frame_{f:06d}.npz',R=R,t=t,object_scale=1.,vertices_camera=camera,head_camera=g['J'],unit_basis=np.eye(3),local_vertices=verts)
  row={'frame_index':f,'head_parent_frame':parents[f],'d':g['d'],'b':g['b'],'a_norm':g['a_norm'],'u_norm':g['u_norm'],'formula_rotation_valid':g['R'] is not None,'object_transform_max_abs_m':residual,'positive_depth_vertices':bool((camera[:,2]>0).all()),'orthogonality_error':float(np.max(np.abs(R.T@R-np.eye(3)))),'determinant':float(np.linalg.det(R)),'host_object_extra_F_or_100':False,**{k:v for k,v in pr.items() if k not in ('R','t')}}
  geomrows.append(row)
 dump(O/'proxy/frames_20261016.json',geomrows)
 model=call('rig_load',lambda:torch.jit.load(asset('rig'),map_location='cpu').eval(),1)
 args=list(model.forward.schema.arguments)[1:]
 assert [a.name for a in args]==['identity_coeffs','model_parameters','face_expr_coeffs','apply_correctives']
 dump(O/'rig/public_forward_schema_20261016.json',{'schema':str(model.forward.schema),'identity_shape':[1,45],'params_shape':[1,204],'expr_shape':[1,72],'apply_correctives':True})
 ti=torch.tensor(shape[None],dtype=torch.float32);expr=torch.zeros((1,72),dtype=torch.float32);checks=[]
 def forward(p):
  with torch.inference_mode():v,j=model(ti,torch.tensor(p[None],dtype=torch.float32),expr,True)
  v,j=v.cpu().numpy(),j.cpu().numpy()
  assert v.shape==(1,18439,3) and j.shape==(1,127,8) and np.isfinite(v).all() and np.isfinite(j).all()
  return v[0].astype(np.float64)*F/100,j[0,:,:3].astype(np.float64)*F/100
 for f in ids:
  a=effective[f];p=a['mhr_params'][0].copy();p[136:]=scales;cam=a['cam_trans'][0].astype(np.float64);q=p.copy();q[:3]+=np.asarray(10*F*cam,np.float32)
  vb,jb=call('rig_forward',lambda:forward(p),64);va,ja=call('rig_forward',lambda:forward(q),64)
  vr=va-vb-cam;jr=ja-jb-cam
  errors={'frame_index':f,'joint22_max_abs_m':float(np.abs(jr[SCORED]).max()),'mesh_max_abs_m':float(np.abs(vr).max()),'mesh_rms_m':float(np.sqrt(np.mean(vr**2))),'all_joints_finite':True,'all_vertices_finite':True,'original_root':p[:3].tolist(),'adapted_root':q[:3].tolist(),'cam_trans':cam.tolist(),'head_parent_frame':parents[f]}
  checks.append(errors)
  np.savez_compressed(O/'rig'/f'frame_{f:06d}.npz',original_params=p,adapted_params=q,baseline_joints=jb,adapter_joints=ja,baseline_vertices=vb,adapter_vertices=va,joint_residual=jr,mesh_residual=vr,scored_ids=SCORED)
  dump(O/'rig/checks_20261016.json',checks)
  assert errors['joint22_max_abs_m']<=1e-4 and errors['mesh_max_abs_m']<=1e-4,'root mapping failed; stop affected production'
  event('rig_pair_completed',**errors)
 assert COUNTS['rig_forward_attempted']==2*len(ids)
 return {'identity_samples':len(S),'hmr_filled':sum(f!=parents[f] for f in ids),'dimension_samples':len(dimids),'rig_pairs':len(checks),'joint22_max_abs_m':max(x['joint22_max_abs_m'] for x in checks),'mesh_max_abs_m':max(x['mesh_max_abs_m'] for x in checks),'contact_status':'unknown','physical_scale_status':'monocular_prior_unknown','head70_rig_entity_equivalence':'unknown'}

def real_commit():
 head=(P/'.git/HEAD').read_text().strip()
 return (P/'.git'/head[5:]).read_text().strip() if head.startswith('ref: ') else head

def stageC():
 import pandas as pd
 sys.path.insert(0,CFG['mesh_site']);sys.path.insert(0,CFG['kit'])
 from tools import pack_reconstruction as pack
 from v2dlb import mhr_submission as ms
 from v2dlb.mesh_common import _mesh_faces
 runtime(['numpy','pandas','pyarrow','trimesh','fast_simplification'])
 official_before={str(p):sha(p) for p in Path(CFG['kit']).rglob('*.py')}
 ids=frame_ids();identity=dict(np.load(O/'identity/fixed_identity_20261016.npz'));mesh=dict(np.load(O/'proxy/canonical_mesh_20261016.npz'))
 fill={r['frame_index']:r['parent_frame'] for r in load(O/'identity/hmr_fill_20261016.json')['records']}
 T=ids[-1]+1;pose=np.full((T,136),np.nan);R=np.full((T,3,3),np.nan);t=np.full((T,3),np.nan);mask=np.zeros(T,bool)
 for f in ids:
  rig=dict(np.load(O/'rig'/f'frame_{f:06d}.npz'));proxy=dict(np.load(O/'proxy'/f'frame_{f:06d}.npz'))
  pose[f]=rig['adapted_params'][:136];R[f]=proxy['R'];t[f]=proxy['t'];mask[f]=True
 assert pose.shape==(T,136) and R.shape==(T,3,3) and t.shape==(T,3) and np.isfinite(pose[ids]).all() and np.isfinite(R[ids]).all() and np.isfinite(t[ids]).all()
 ep=O/'episodes'/f'episode_{MAN["episode"]:06d}.npz'
 np.savez_compressed(ep,pose=pose,scales=identity['scales'],shape=identity['shape'],object_rotation=R,object_translation=t,object_scale=np.array(1.))
 np.savez_compressed(O/'episodes/completed_mask_20261016.npz',completed_mask=mask,frame_indices=ids)
 obj=ep.with_name(ep.stem+'_object.obj')
 with obj.open('w') as f:
  f.write('# Input-driven box; local metres, centre origin; not a measured surface\n')
  for v in mesh['vertices']:f.write('v '+' '.join(format(float(x),'.17g') for x in v)+'\n')
  for face in mesh['faces']:f.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
 assert not any(obj.with_suffix(s).exists() for s in ('.glb','.ply','.stl','.off'))
 sample=pd.read_parquet(asset('sample'),columns=['row_id']);parts=sample.row_id.str.split('_',expand=True)
 e=parts[1].astype(int);fr=parts[2].astype(int)
 local=sample[(e==MAN['episode']) & ((fr==999999)|fr.isin(ids))].reset_index(drop=True)
 assert len(local)==8231+50*len(ids)==9831 and local.row_id.is_unique
 samplepath=O/'pack/local32_row_ids_not_for_submission.parquet';local.to_parquet(samplepath,index=False)
 marker='local-git:'+MAN['commit'];assert real_commit()==MAN['commit']
 official_pattern_accepts=bool(__import__('re').match(pack.rs.CODE_COMMIT_PATTERN,marker));assert not official_pattern_accepts
 def write_local(rows,sample,commit,out):
  assert __import__('re').fullmatch(r'local-git:[0-9a-f]{40}',commit) and commit==marker and commit=='local-git:'+real_commit()
  assert rows.row_id.is_unique and sample.row_id.is_unique and len(rows)==len(sample)
  merged=sample.merge(rows,on='row_id',how='left',validate='one_to_one',sort=False)
  assert merged.row_id.tolist()==sample.row_id.tolist() and not merged[['x','y','z']].isna().any().any() and np.isfinite(merged[['x','y','z']].to_numpy()).all()
  merged['code_commit_url']=commit;merged[['row_id','x','y','z','code_commit_url']].to_parquet(out,index=False)
 stock_write=pack.write;pack.write=write_local
 out=O/'pack/track1_local32_not_for_submission.parquet'
 try:pack.pack_track1(argparse.Namespace(episodes=O/'episodes',sample=samplepath,commit=marker,out=out))
 finally:pack.write=stock_write
 result=pd.read_parquet(out);assert result.columns.tolist()==['row_id','x','y','z','code_commit_url'] and result.row_id.tolist()==local.row_id.tolist() and result.row_id.is_unique and np.isfinite(result[['x','y','z']].to_numpy()).all() and result.code_commit_url.unique().tolist()==[marker]
 ps=result.row_id.str.split('_',expand=True)
 block=result[['x','y','z']].copy();block['episode_index']=ps[1].astype(int);block['frame_index']=ps[2].astype(int);block['array_name']=ps[3].astype(int).map(dict(enumerate(ms.array_order())));block['point_index']=ps[4].astype(int)
 arrays=ms._to_arrays(block,['x','y','z'])
 poses=ms._unpack(arrays['mhr_pose'],136,'mhr_pose');scales=ms._unpack(arrays['mhr_scales'],68,'mhr_scales');shapes=ms._unpack(arrays['mhr_shape'],45,'mhr_shape')
 assert np.array_equal(poses,pose[ids]) and np.array_equal(scales,identity['scales']) and np.array_equal(shapes,identity['shape'])
 projected=ms._rotations(arrays['object_rotation']);scale=ms._object_scale(arrays['object_scale'])
 assert np.array_equal(arrays['object_rotation'],R[ids]) and np.array_equal(arrays['object_translation'][:,0],t[ids]) and scale==1
 vv=arrays['object_mesh_vertices'];ff=_mesh_faces(arrays['object_mesh_faces'],len(vv));tri=vv[ff];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
 assert vv.shape==(4096,3) and ff.shape==(4096,3) and np.isfinite(areas).all() and areas.sum()>0 and np.count_nonzero(areas)==12
 assert np.array_equal(vv[8:],np.repeat(vv[:1],4088,axis=0)) and np.array_equal(ff[12:],np.zeros((4084,3),int))
 end_to_end=[]
 # Official welding may reorder vertices; compare geometry as sorted coordinate sets.
 assert np.array_equal(np.unique(vv,axis=0),np.unique(mesh['vertices'],axis=0))
 for f in ids:
  i=ids.index(f);pr=dict(np.load(O/'proxy'/f'frame_{f:06d}.npz'))
  official_path=(arrays['object_rotation'][i]@vv.T).T*scale+arrays['object_translation'][i,0]
  direct=vv@pr['R'].T+pr['t'];err=float(np.abs(official_path-direct).max());assert err<=1e-10;end_to_end.append(err)
 assert all(sha(p)==h for p,h in official_before.items()),'official disk source modified'
 validation={'numeric_index_mesh_pack':'passed','rows':len(result),'frames':ids,'episode':MAN['episode'],'columns':result.columns.tolist(),'unique_and_ordered':True,'finite':True,'tails_zero':True,'pose136_scales68_shape45_roundtrip':True,'completed_count':int(mask.sum()),'T':T,'unprocessed_nan_and_false':bool(np.isnan(pose[~mask]).all() and not mask[~mask].any()),'R_row_encoding':True,'R_generation_orth_max':float(np.abs(R[ids].transpose(0,2,1)@R[ids]-np.eye(3)).max()),'R_official_projection_max_change':float(np.abs(projected-R[ids]).max()),'object_scale':scale,'mesh_vertices':4096,'mesh_faces':4096,'nonzero_area_faces':int(np.count_nonzero(areas)),'mesh_total_area_m2':float(areas.sum()),'object_transform_max_abs_m':max(end_to_end),'official_disk_source_unchanged':True,'official_writer':'not_used_original_terminal_write_local_L1','official_code_commit_url_format':'failed_local_marker_expected','remote_commit_accessibility':'unpublished','full_roster':'partial_1_episode_32_frames_of_30_9877','kaggle_scoring':'not_run','public_row':'not_run','provenance_kind':'local_git','submission_eligible':False,'code_commit_url':marker,'commit':MAN['commit'],'npz_sha256':sha(ep),'obj_sha256':sha(obj),'sample_sha256':sha(samplepath),'parquet_sha256':sha(out),'official_packer_sha256':sha(Path(CFG['kit'])/'tools/pack_reconstruction.py')}
 dump(O/'pack/validation_20261016.json',validation);dump(O/'pack/track1_local32_not_for_submission.sidecar_20261016.json',validation)
 render_overlays(ids)
 return validation

def render_overlays(ids):
 import cv2
 cv2.setNumThreads(1)
 for f in (50,60,70,81):
  assert f in ids
  canvas=cv2.imread(str(O/'source'/f'frame_{f:06d}.png'));tr=load(O/'hmr'/f'transform_{f:06d}.json');K=np.asarray(tr['K']);g=dict(np.load(O/'proxy'/f'frame_{f:06d}.npz'));rig=dict(np.load(O/'rig'/f'frame_{f:06d}.npz'));rec=load(O/'detector'/f'decision_{f:06d}.json')
  def project(x):
   z=x[:,2];return x[:,:2]/z[:,None]*K[0,0]+K[:2,2]
  def valid(pt):return np.isfinite(pt).all() and np.abs(pt).max()<1e6
  x1,y1,x2,y2=map(int,rec['bbox']);cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,200,255),3)
  for points,color,radius in [(g['head_camera'],(80,255,80),3),(rig['adapter_joints'][SCORED],(255,160,40),5)]:
   for pt in project(points):
    if valid(pt):cv2.circle(canvas,tuple(np.rint(pt).astype(int)),radius,color,-1)
  uv=project(g['vertices_camera'])
  for a,b in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):
   if valid(uv[a]) and valid(uv[b]):cv2.line(canvas,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),(255,80,230),3)
  raw=dict(np.load(O/'hmr'/f'raw_{f:06d}.npz'));norm=raw['joints_2d'][0];sx,sy,ss=tr['square_float'];head_uv=(norm+1)*.5*ss+np.array([sx,sy]);proj_uv=project(g['head_camera']);rmse=float(np.sqrt(np.mean((head_uv-proj_uv)**2)))
  dump(O/'overlays'/f'projection_{f:06d}.json',{'frame_index':f,'virtual_K':K.tolist(),'head2d_vs_head3d_projection_rmse_pixels':rmse,'rig127_head70_direct_index_comparison':'not_valid_different_semantics','quality':'contact_pixel_support_real_scale_unknown'})
  cv2.putText(canvas,f'{f} {rec["branch"]} | green head70 / blue rig22 / pink proxy | quality UNKNOWN',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
  assert cv2.imwrite(str(O/'overlays'/f'overlay_{f:06d}.jpg'),canvas)

def main():
 global O,STAGE,CFG,MAN
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['A','B','C'],required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();O=args.output.resolve();STAGE=args.stage;CFG=load(P/'config_20261016.json');MAN=load(O/'run_manifest_20261016.json')
 assert real_commit()==MAN['commit'] and MAN['config_sha256']==sha(P/'config_20261016.json')
 def alarm(*_):raise TimeoutError('stage wall budget exhausted')
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,CFG['limits'][STAGE]['wall_s'])
 allowed={a['path'] for a in MAN['assets'].values()};allowed.add(CFG['mesh_site']);allowed.add(CFG['kit'])
 root=Path(CFG['root']);prefixes=[str(O)+'/',str(P)+'/',CFG['mesh_site']+'/',CFG['kit']+'/']
 def audit(kind,args):
  if kind in ('socket.connect','socket.bind','subprocess.Popen','os.system'):raise RuntimeError('worker external action denied '+kind)
  if kind=='open' and isinstance(args[0],(str,bytes)):
   path=str(Path(os.fsdecode(args[0])).absolute())
   if path.startswith(str(root)+'/'):
    if path not in allowed and not any(path.startswith(x) for x in prefixes):raise RuntimeError('non-allowlisted project input '+path)
    READS.add(path)
 sys.addaudithook(audit);event('worker_start',stage=STAGE,commit=MAN['commit'],pid=os.getpid())
 status='failed';details={}
 try:
  details={'A':stageA,'B':stageB,'C':stageC}[STAGE]();status='completed'
 except BaseException as e:
  details={'error':repr(e),'traceback':traceback.format_exc()};event('failure',**details);raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  dump(O/'logs'/f'{STAGE}_receipt_20261016.json',{'stage':STAGE,'status':status,'counts':dict(COUNTS),'stage_seconds':dict(TIMES),'elapsed_worker_s':time.perf_counter()-START,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'details':details,'project_reads':sorted(READS),'audit_scope':'Python audit plus explicit native input paths; not syscall trace','network_actions':0,'warmups':0,'retries':0,'commit':MAN['commit']})
if __name__=='__main__':main()
