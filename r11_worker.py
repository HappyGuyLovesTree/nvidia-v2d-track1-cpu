"""R011 serial episode stages. No automatic forward retry; durable safe-point recovery."""
from pathlib import Path
from fractions import Fraction
import argparse, collections, hashlib, io, json, math, os, signal, sys, time, traceback
import numpy as np
from common import sha, load, candidate, nearest, proxy_geometry, box_mesh
from r11_geometry import complete_proxy_source as complete_proxy
from tracking import LowScoreTracker, validate_raw, digest
from r11_state import *
from r11_inputs import *
EXPECTED={'mhr_params':(1,204),'shape_params':(1,45),'cam_trans':(1,3),'joints_2d':(1,70,2),'joints_3d':(1,70,3)}
SCORED=np.array([1,2,18,35,3,19,36,4,20,37,8,24,110,74,38,113,75,39,76,40,78,42]);F=np.array([1.,-1.,-1.])

def valid_raw(raw,keys=EXPECTED):
    return set(raw)==set(EXPECTED) and all(raw[k].shape==EXPECTED[k] and raw[k].dtype==np.float32 and np.isfinite(raw[k]).all() for k in keys)
def end_index(ledger):
    out={}
    if ledger.journal.exists():
        with ledger.journal.open('rb') as f:
            while True:
                offset=f.tell();b=f.readline()
                if not b:break
                require(b.endswith(b'\n'),'torn call journal: preserve and investigate')
                row=json.loads(b)
                if row['kind']=='call_end':
                    out[(row['episode'],row['frame'],row['model_kind'],row['stage'])]=out[(row['episode'],row['frame'],row['model_kind'])]=({'path':str(ledger.journal.resolve()),'offset':offset,'length':len(b),'sha256':hashlib.sha256(b).hexdigest()},row)
    return out

def prepare_transform(rgb,bbox):
    from vendor.preprocess import _preprocess
    h,w=rgb.shape[:2];crop,sx,sy,ss,cliff=_preprocess(rgb,np.asarray(bbox,np.float32),h,w)
    require(crop.shape==(3,224,224) and crop.dtype==np.float32 and cliff.shape==(3,) and cliff.dtype==np.float32 and np.isfinite(crop).all() and np.isfinite(cliff).all(),'invalid HMR input')
    ix1,iy1,ix2,iy2=math.floor(sx),math.floor(sy),math.ceil(sx+ss),math.ceil(sy+ss)
    transform={'bbox':bbox,'square_float':[sx,sy,ss],'integer_extent':[ix1,iy1,ix2,iy2],'padding_ltrb':[max(0,-ix1),max(0,-iy1),max(0,ix2-w),max(0,iy2-h)],'source_extent':[max(0,ix1),max(0,iy1),min(w,ix2),min(h,iy2)],'wrapper_crop_to_original':[[ss/224,0,sx],[0,ss/224,sy],[0,0,1]],'integer_resize_to_original':[[(ix2-ix1)/224,0,ix1],[0,(iy2-iy1)/224,iy1],[0,0,1]],'cliff':cliff.tolist(),'K':[[math.hypot(w,h),0,w/2],[0,math.hypot(w,h),h/2],[0,0,1]],'K_source':'virtual_hypot_not_measured'}
    return {'image':crop[None],'cliff_cond':cliff[None]},transform

def restore_record(record,tracker,identity,nh,context,parents,inputs,previous,ep):
    require(record['episode_input_sha256']==digest(ep) and record['numeric_hash']==nh and record['producer']==identity['producer_commit'],'frame producer/input mismatch')
    require(record['previous_frame_receipt']==previous and record['input_identity']==inputs,'frame input or parent chain mismatch')
    decision=tracker.step(context['previous_frame'],record['decision']['candidates'],context['width'],context['height'],episode=context['episode'],source_segment=context['source_segment'],pts=context['previous_pts'],time_base=context['time_base'],input_parents=parents)
    require(digest(decision)==digest(record['decision']),'restored decision differs from continuous processing')
    require(record['tracker_after']['state_sha256']==decision['tracker_after_sha256'],'tracker decision/state mismatch')
    restored=LowScoreTracker.restore_state(record['tracker_after'],identity=identity,numeric_config_hash=nh,expected_state_sha256=decision['tracker_after_sha256'],expected_input_parents=parents,expected_context=context)
    detend=read_line_ref(record['detector_call_end']);require(detend['kind']=='call_end' and detend['model_kind']=='detector' and detend['episode']==ep['episode'] and detend['frame']==record['frame'] and detend['outcome']=='returned' and detend['input_identity']==digest({k:v for k,v in inputs.items() if k!='hmr_tensor_hashes'}),'invalid detector closure')
    end=read_line_ref(record['hmr_call_end']);require(end['kind']=='call_end' and end['episode']==ep['episode'] and end['frame']==record['frame'] and end['model_kind']=='hmr' and end['input_identity']==digest(inputs['hmr_tensor_hashes']),'wrong closed HMR evidence')
    if record['hmr_outcome']=='hmr_valid':
        raw=read_npz(verify_ref(record['raw']));require(valid_raw(raw) and end['outcome']=='returned' and end['output_identity']==arrays_hash(raw),'invalid HMR cache')
    else:
        require(record['hmr_outcome']=='hmr_failed' and record['failure'],'not a known failed terminal')
        if record.get('raw'):
            raw=read_npz(verify_ref(record['raw']));require(not valid_raw(raw) and end['output_identity']==arrays_hash(raw),'bad raw evidence mismatch')
        else:require(end['outcome']=='failed','exception evidence missing')
    return restored

def save_audit_frame(er,fid,rgb,audit,writer,detection_path):
    if fid in audit:
        import cv2
        source=er/'audit'/f'{fid:06d}_source.jpg'
        if not source.exists():
            ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,94]);require(ok,'audit encoding failed')
            writer.bytes(source,encoded.tobytes(),'images')
        detout=er/'audit'/f'{fid:06d}_detector.npz'
        if detection_path.exists() and not detout.exists():
            d=load(detection_path);writer.bytes(detout,verify_ref(d['raw']).read_bytes(),'A')
    if detection_path.exists():
        writer.remove(Path(load(detection_path)['raw']['path']),'A');writer.remove(detection_path,'A')

def stage_a(root,er,ep,cfg,man,ledger,writer,owner):
    ledger.assert_no_unknown()
    import cv2,av,onnxruntime as ort
    from vendor.nanodet import NanoDet
    from vendor.letterbox import letterbox,unletterbox
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    rt=runtime_identity(['numpy','cv2','av','onnxruntime']);identity=numeric_identity(cfg,man,rt,ep);nh=numerical_hash(cfg)
    rp=er/'runtime_A.json'
    if rp.exists():require(load(rp)==rt,'actual runtime changed')
    else:writer.json(rp,rt)
    pts=verify_episode(ep,cfg);required=set(ep['required']);audit={f for e,f in man['audit']['pairs'] if e==ep['episode']}
    tracker=LowScoreTracker(identity=identity,numeric_config_hash=nh);store=FrameStore(er,writer);records=store.records();ends=end_index(ledger)
    detector=sess=None;previous=None;seen=set();decoded=0;start=time.monotonic()
    # Pure preprocessing metadata. No DNN constructor/load for already committed frames.
    proto=NanoDet.__new__(NanoDet);proto.mean=np.array([103.53,116.28,123.675],np.float32).reshape(1,1,3);proto.std=np.array([57.375,57.12,58.395],np.float32).reshape(1,1,3)
    with av.open(ep['video']['path']) as container:
        stream=next(s for s in container.streams.video if s.index==ep['stream']['index']);stream.codec_context.thread_count=1;stream.codec_context.thread_type='NONE'
        require(str(Fraction(stream.time_base))==str(Fraction(ep['stream']['time_base'])) and stream.codec_context.name==ep['stream']['codec_name'],'stream identity mismatch')
        for fid,frame in enumerate(container.decode(stream)):
            ledger.check_time(model=True);decoded+=1
            require(fid<len(pts['frame_map']) and frame.pts==pts['frame_map'][fid]['pts'] and Fraction(frame.time_base)==Fraction(ep['stream']['time_base']),'actual decoded ordinal/PTS mismatch')
            require((frame.width,frame.height)==(ep['stream']['width'],ep['stream']['height']),'actual dimensions changed')
            if fid not in required:continue
            seen.add(fid);rgb=np.ascontiguousarray(cv2.cvtColor(frame.to_ndarray(format='bgr24'),cv2.COLOR_BGR2RGB));h,w=rgb.shape[:2]
            img,scale=letterbox(rgb);blob=proto.pre_process(img)
            require(blob.shape==(1,3,416,416) and blob.dtype==np.float32 and np.isfinite(blob).all(),'invalid detector input')
            source=int(ep['source_map'][str(fid)]);base={'episode':ep['episode'],'source':source,'frame':fid,'pts':int(frame.pts),'time_base':str(Fraction(frame.time_base)),'stream':stream.index,'width':w,'height':h,'video_sha256':ep['video']['sha256'],'whole_pts_sha256':ep['whole_pts']['sha256'],'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'blob_sha256':array_hash(blob)}
            existing=records.get(fid) or store.orphan(fid)
            detection_path=er/'detection'/f'{fid:06d}.json'
            if existing:
                rawhash=existing['detector_raw_sha256'];cs=existing['decision']['candidates'];det_end=existing['detector_call_end']
            elif detection_path.exists():
                det=load(detection_path);require(det['input_identity']==base,'pending detection input mismatch');verify_ref(det['raw']);dr=read_npz(det['raw']);validate_raw(dr)
                end=read_line_ref(det['call_end']);require(end['output_identity']==arrays_hash(dr) and end['input_identity']==digest(base) and end['episode']==ep['episode'] and end['frame']==fid and end['model_kind']=='detector' and end['outcome']=='returned','pending detection closure mismatch');rawhash=det['raw']['sha256'];cs=det['candidates'];det_end=det['call_end']
            else:
                require(not ledger.unknown_for(ep['episode'],fid),'unknown call requires explicit investigation; never guessed/retried')
                require((ep['episode'],fid,'detector') not in ends,'closed detector result absent; no automatic repeated forward')
                if detector is None:
                    detector,_=ledger.invoke('dnn_load',lambda:NanoDet(cfg['assets']['nanodet']['path'],.10,.60,cv2.dnn.DNN_BACKEND_OPENCV,cv2.dnn.DNN_TARGET_CPU),episode=ep['episode'],frame=None,attempt=owner.attempt,stage='A',n=len(required))
                def detect():detector.net.setInput(blob);return detector.net.forward(detector.net.getUnconnectedOutLayersNames())
                detraw,det_end=ledger.invoke('detector',detect,episode=ep['episode'],frame=fid,attempt=owner.attempt,stage='A',n=len(required),output_identity=lambda a:arrays_hash({f'output_{i}':v for i,v in enumerate(a)}),input_identity=digest(base))
                validate_raw(detraw);dr={f'output_{i}':v for i,v in enumerate(detraw)}
                rawref=writer.npz(er/'scratch'/f'detector_{fid:06d}.npz','A',**dr);rawhash=rawref['sha256']
                preds=detector.post_process(detraw);cs=[];top,left,newh,neww=scale
                if preds.size==0:preds=np.empty((0,6))
                require(preds.ndim==2 and preds.shape[1]==6,'invalid postprocessed detector shape')
                for index,row in enumerate(preds):
                    box=row[:4];valid=np.isfinite(row).all();inv=[float((box[0]-left)*w/neww),float((box[1]-top)*h/newh),float((box[2]-left)*w/neww),float((box[3]-top)*h/newh)] if valid else None
                    xy=unletterbox(box,(h,w),scale) if valid else [float('nan')]*4
                    cs.append(candidate(index,xy,float(row[4]),int(row[5]) if valid else -1,w,h,letterbox_xyxy=box.tolist() if valid else None,inverse_float_unclipped=inv,paired_inverse_int32=np.asarray(xy,dtype=np.int32).tolist() if valid else None))
                writer.json(detection_path,{'input_identity':base,'raw':rawref,'candidates':cs,'call_end':det_end},'A',True)
            parents={'RGB':base['rgb_sha256'],'blob':base['blob_sha256'],'raw':rawhash,'wholePTS':ep['whole_pts']['sha256']}
            context={'episode':ep['episode'],'source_segment':source,'previous_frame':fid,'previous_pts':int(frame.pts),'time_base':str(Fraction(frame.time_base)),'width':w,'height':h}
            if existing:
                tensors,transform=prepare_transform(rgb,existing['decision']['bbox']);inputs={**base,'hmr_tensor_hashes':{k:array_hash(v) for k,v in tensors.items()}}
                tracker=restore_record(existing,tracker,identity,nh,context,parents,inputs,previous,ep)
                require(existing['transform']==transform,'transform identity mismatch')
                if fid not in records:store.append_commit(fid,ref(er/'frames'/f'{fid:06d}.json'));records[fid]=existing
                previous=digest(existing)
                writer.remove(er/'prepared'/f'{fid:06d}.json','A')
                save_audit_frame(er,fid,rgb,audit,writer,detection_path)
                continue
            decision=tracker.step(fid,cs,w,h,episode=ep['episode'],source_segment=source,pts=int(frame.pts),time_base=str(Fraction(frame.time_base)),input_parents=parents)
            tensors,transform=prepare_transform(rgb,decision['bbox']);inputs={**base,'hmr_tensor_hashes':{k:array_hash(v) for k,v in tensors.items()}}
            prep={'schema':'r11_frame_v1','status':'frame_committed','episode':ep['episode'],'frame':fid,'source':source,'episode_input_sha256':digest(ep),'producer':man['code']['commit'],'numeric_hash':nh,'runtime_sha256':digest(rt),'input_identity':inputs,'detector_raw_sha256':rawhash,'detector_call_end':det_end,'decision':decision,'tracker_after':tracker.export_state(),'transform':transform,'previous_frame_receipt':previous,'cache_layers_omitted':['RGB_tensor','detector_blob','HMR_inputs']+([] if fid in audit else ['detector_raw']),'attempt':owner.attempt}
            pp=er/'prepared'/f'{fid:06d}.json'
            if pp.exists():
                old=load(pp);require(digest({k:v for k,v in old.items() if k!='attempt'})==digest({k:v for k,v in prep.items() if k!='attempt'}),'prepared frame identity changed');prep=old
            else:writer.json(pp,prep,'A',True)
            raw_path=er/'hmr'/f'{fid:06d}.npz';failure=None;rawref=None
            closed=ends.get((ep['episode'],fid,'hmr'))
            if closed:
                endref,end=closed
                require(end['input_identity']==arrays_hash(tensors) and end['episode']==ep['episode'] and end['frame']==fid and end['model_kind']=='hmr','closed HMR input parent mismatch')
                if end['outcome']=='failed':failure={'kind':'exception','error':end['error']}
                else:
                    require(raw_path.exists(),'closed HMR result absent; cannot repeat known call');raw=read_npz(raw_path);require(arrays_hash(raw)==end['output_identity'],'closed raw hash conflict');rawref=ref(raw_path)
                    if not valid_raw(raw):failure={'kind':'schema_or_nonfinite','checks':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(np.isfinite(v).all())} for k,v in raw.items()}}
            else:
                require(not ledger.unknown_for(ep['episode'],fid),'unclosed HMR stays unknown; no guessed recovery')
                if sess is None:
                    so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1;so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
                    sess,_=ledger.invoke('ort_session_load',lambda:ort.InferenceSession(cfg['assets']['hmr']['path'],sess_options=so,providers=['CPUExecutionProvider']),episode=ep['episode'],frame=None,attempt=owner.attempt,stage='A',n=len(required))
                    require(sess.get_providers()==['CPUExecutionProvider'],'provider mismatch')
                    nodes={x.name:x for x in sess.get_inputs()};require(set(nodes)==set(tensors) and all(x.type=='tensor(float)' and x.shape[1:]==list(tensors[k].shape[1:]) for k,x in nodes.items()),'model input schema mismatch')
                    require({x.name for x in sess.get_outputs()}==set(EXPECTED),'model output schema mismatch')
                try:
                    vals,endref=ledger.invoke('hmr',lambda:sess.run(list(EXPECTED),tensors),episode=ep['episode'],frame=fid,attempt=owner.attempt,stage='A',n=len(required),output_identity=lambda v:arrays_hash(dict(zip(EXPECTED,v))),input_identity=arrays_hash(tensors))
                except ClosedCallError as e:failure={'kind':'exception','error':str(e)};endref=ledger.failed_end(e.call_id)
                else:
                    raw=dict(zip(EXPECTED,vals));rawref=writer.npz(raw_path,'A',**raw)
                    if not valid_raw(raw):failure={'kind':'schema_or_nonfinite','checks':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'finite':bool(np.isfinite(v).all())} for k,v in raw.items()}}
            record={**prep,'hmr_outcome':'hmr_failed' if failure else 'hmr_valid','failure':failure,'raw':rawref,'hmr_call_end':endref}
            store.commit(record);records[fid]=record;previous=digest(record)
            writer.remove(pp,'A')
            save_audit_frame(er,fid,rgb,audit,writer,detection_path)
    require(decoded==ep['decoded_frame_count'] and seen==required,'full decode/required coverage mismatch')
    verify_episode(ep,cfg);ledger.assert_no_unknown()
    return {'status':'completed','decoded_frames':decoded,'frames':sorted(records),'hmr_valid':[f for f,r in records.items() if r['hmr_outcome']=='hmr_valid'],'hmr_failed':[f for f,r in records.items() if r['hmr_outcome']=='hmr_failed'],'branches':dict(collections.Counter(r['decision']['branch'] for r in records.values())),'elapsed_s':time.monotonic()-start,'frame_journal':ref(store.journal),'runtime':ref(rp)}

def stage_b(root,er,ep,cfg,man,ledger,writer,owner):
    ledger.assert_no_unknown()
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    rt=runtime_identity(['numpy','torch']);rp=er/'runtime_B.json'
    if rp.exists():require(load(rp)==rt,'rig runtime changed')
    else:writer.json(rp,rt)
    records=FrameStore(er,writer).records();ids=ep['required'];require(set(records)==set(ids),'B requires all A terminal frames')
    raw={f:read_npz(verify_ref(r['raw'])) for f,r in records.items() if r.get('raw')}
    # Identity S deliberately uses the original three-output shape/finite rule.
    S=[f for f,a in raw.items() if all(k in a and a[k].shape==EXPECTED[k] and a[k].dtype==np.float32 and np.isfinite(a[k]).all() for k in ('mhr_params','shape_params','cam_trans'))]
    require(S,'no episode identity estimate');scales=np.median(np.stack([raw[f]['mhr_params'][0,136:] for f in S]),axis=0).astype(np.float32);shape=np.median(np.stack([raw[f]['shape_params'][0] for f in S]),axis=0).astype(np.float32)
    parents={};good=[f for f,r in records.items() if r['hmr_outcome']=='hmr_valid'];geoms={}
    for source in ep['sources']:
        valid=[f for f in source if f in good];require(valid,'source has no valid HMR parent; partial, never borrow across gap')
        for f in source:parents[f]=f if f in valid else nearest(f,valid)
    effective={f:raw[parents[f]] for f in ids};geoms={f:proxy_geometry(a) for f,a in effective.items()}
    dims=[f for f,g in geoms.items() if np.isfinite([g['d'],g['b']]).all() and min(g['d'],g['b'])>0];require(dims,'no input-driven episode mesh dimensions')
    D=np.array([np.median([geoms[f]['d'] for f in dims]),np.median([geoms[f]['b'] for f in dims]),np.median([geoms[f]['b'] for f in dims])]);proxies={}
    for source in ep['sources']:
        _,_,pr=complete_proxy({f:geoms[f] for f in source},dimensions=D);proxies.update(pr)
    verts,faces=box_mesh(D)
    plan={'schema':'r11_B_plan_v1','producer':man['code']['commit'],'episode_input':digest(ep),'frames':{str(f):digest(records[f]) for f in ids},'scales':scales.tolist(),'shape':shape.tolist(),'identity_samples':S,'dimensions':D.tolist(),'dimension_samples':dims,'parents':{str(f):{'frame':parents[f],'raw_sha256':records[parents[f]]['raw']['sha256'],'reason':'original_valid' if f==parents[f] else 'nearest_valid_same_source'} for f in ids},'runtime_sha256':digest(rt),'numeric_hash':numerical_hash(cfg)}
    ph=digest(plan);br=er/'B'/ph
    if (br/'plan.json').exists():require(load(br/'plan.json')==plan,'B parent mismatch')
    else:writer.json(br/'plan.json',plan,'B',True)
    writer.json(er/'B_plan_latest.json',{'path':str(br/'plan.json'),'sha256':sha(br/'plan.json'),'plan_hash':ph})
    writer.npz(br/'identity.npz','B',scales=scales,shape=shape,source_frames=np.array(S));writer.npz(br/'mesh.npz','B',vertices=verts,faces=faces,dimensions=D)
    model=None;ti=expr=None;checked=[];fullmesh={f for e,f in man['audit']['full_mesh_pairs'] if e==ep['episode']};ends=end_index(ledger)
    def forward(p):
        with torch.inference_mode():v,j=model(ti,torch.tensor(p[None],dtype=torch.float32),expr,True)
        v,j=v.cpu().numpy(),j.cpu().numpy();require(v.shape==(1,18439,3) and j.shape==(1,127,8) and np.isfinite(v).all() and np.isfinite(j).all(),'invalid rig output')
        return {'vertices':v[0].astype(np.float64)*F/100,'joints':j[0,:,:3].astype(np.float64)*F/100}
    for f in ids:
        ledger.check_time(model=True);receipt=br/'frames'/f'{f:06d}.json';output=br/'frames'/f'{f:06d}.npz'
        if receipt.exists():
            r=load(receipt);require(r['plan_hash']==ph and r['frame']==f,'rig plan mismatch');verify_ref(r['output']);require(r['checks']['joint22_max_abs_m']<=1e-4 and r['checks']['mesh_max_abs_m']<=1e-4,'invalid rig receipt')
            for rr in r['call_ends']:require(read_line_ref(rr)['kind']=='call_end','unclosed rig')
            for label in ('baseline','adapted'):
                writer.remove(br/'inflight'/f'{f:06d}_{label}.npz','B');writer.remove(br/'inflight'/f'{f:06d}_{label}.json','B')
            checked.append(f);continue
        a=effective[f];p=a['mhr_params'][0].copy();p[136:]=scales;cam=a['cam_trans'][0].astype(np.float64);q=p.copy();q[:3]+=np.asarray(10*F*cam,np.float32)
        pair=[];callends=[]
        for label,params in (('baseline',p),('adapted',q)):
            cp=br/'inflight'/f'{f:06d}_{label}.json';data=cp.with_suffix('.npz');callinput=digest({'params':array_hash(params),'shape':array_hash(shape[None]),'plan':ph})
            if cp.exists():
                cr=load(cp);require(cr['params_hash']==array_hash(params) and cr['plan_hash']==ph,'inflight rig parent mismatch');values=read_npz(verify_ref(cr['output']));end=read_line_ref(cr['call_end']);require(end['outcome']=='returned' and end['output_identity']==arrays_hash(values) and end['input_identity']==callinput and end['episode']==ep['episode'] and end['frame']==f and end['stage']=='B:'+label,'rig closure mismatch');rr=cr['call_end']
            else:
                require(not ledger.unknown_for(ep['episode'],f),'unknown rig frame: investigate before new forward')
                # A known completed but unpersisted forward cannot be silently run again.
                if (ep['episode'],f,'rig','B:'+label) in ends:raise ValueError('uncommitted rig call requires explicit investigation')
                if model is None:
                    model,_=ledger.invoke('rig_load',lambda:torch.jit.load(cfg['assets']['rig']['path'],map_location='cpu').eval(),episode=ep['episode'],frame=None,attempt=owner.attempt,stage='B',n=len(ids))
                    require([a.name for a in list(model.forward.schema.arguments)[1:]]==['identity_coeffs','model_parameters','face_expr_coeffs','apply_correctives'],'rig schema mismatch')
                    ti=torch.tensor(shape[None],dtype=torch.float32);expr=torch.zeros((1,72),dtype=torch.float32)
                values,rr=ledger.invoke('rig',lambda:forward(params),episode=ep['episode'],frame=f,attempt=owner.attempt,stage='B:'+label,n=len(ids),output_identity=arrays_hash,input_identity=callinput)
                dr=writer.npz(data,'B',**values);writer.json(cp,{'plan_hash':ph,'params_hash':array_hash(params),'output':dr,'call_end':rr},'B',True)
            pair.append(values);callends.append(rr)
        vb,jb=pair[0]['vertices'],pair[0]['joints'];va,ja=pair[1]['vertices'],pair[1]['joints'];vr=va-vb-cam;jr=ja-jb-cam
        checks={'joint22_max_abs_m':float(np.abs(jr[SCORED]).max()),'mesh_max_abs_m':float(np.abs(vr).max()),'mesh_rms_m':float(np.sqrt(np.mean(vr**2)))}
        require(checks['joint22_max_abs_m']<=1e-4 and checks['mesh_max_abs_m']<=1e-4,'root mapping failed')
        pr=proxies[f];camera=verts@pr['R'].T+pr['t'];require(np.abs(camera-((pr['R']@verts.T).T+pr['t'])).max()<=1e-10,'object path mismatch')
        outref=writer.npz(output,'B',original_params=p,adapted_params=q,adapter_joints=ja,joint_residual=jr,R=pr['R'],t=pr['t'],head_camera=geoms[f]['J'],vertices_camera=camera)
        meshref=writer.npz(br/'mesh_audit'/f'{f:06d}.npz','mesh',baseline_vertices=vb,adapter_vertices=va,mesh_residual=vr,baseline_joints=jb,adapter_joints=ja) if f in fullmesh else None
        writer.json(receipt,{'frame':f,'source':ep['source_map'][str(f)],'plan_hash':ph,'output':outref,'checks':checks,'call_ends':callends,'full_mesh':meshref,'hmr_fill':plan['parents'][str(f)],'proxy':{k:v for k,v in pr.items() if k not in ('R','t')},'quality':'contact_real_scale_surface_unknown'},'B',True)
        checked.append(f)
        # Complete pair is checked over all vertices. Release transient full meshes.
        for label in ('baseline','adapted'):
            writer.remove(br/'inflight'/f'{f:06d}_{label}.npz','B');writer.remove(br/'inflight'/f'{f:06d}_{label}.json','B')
    ledger.assert_no_unknown()
    return {'status':'completed','plan':ref(br/'plan.json'),'identity':ref(br/'identity.npz'),'mesh':ref(br/'mesh.npz'),'rig_checked':checked,'hmr_filled':[f for f in ids if f!=parents[f]],'identity_samples':S,'runtime':ref(rp),'quality':'unknown'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['A','B','C'],required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--episode',type=int,required=True);args=ap.parse_args()
    root=args.root.resolve();man=load(root/'run_manifest.json');cfg=load(man['config']['path']);verify_ref(man['config']);verify_code(man)
    if man.get('schema')=='r12_run_v1':
        from r12_session import runtime_config, child_bridge
        cfg=runtime_config(man,root);child_bridge(root,cfg,args.episode,args.stage)
    ep=load(root/'inputs'/f'episode_{args.episode:06d}.json');verify_episode(ep,cfg)
    for v in cfg['assets'].values():verify_ref(v)
    er=root/'episode_runs'/f'episode_{args.episode:06d}';ledger=Ledger(root,cfg);writer=Writer(ledger)
    def stop(*_):raise BudgetStop('supervised_stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,min(cfg['stage_limits'][args.stage]['wall_s'],max(.001,datetime.datetime.fromisoformat(cfg['deadlines'].get('candidate_stop',cfg['deadlines']['candidate_freeze'])).timestamp()-time.time())))
    allowed={str(Path(a['path']).resolve()) for a in cfg['assets'].values()}|{str(Path(ep['video']['path']).resolve()),str(Path(ep['whole_pts']['path']).resolve()),str(Path(man['config']['path']).resolve())}
    prefixes=[str(p.resolve())+'/' for p in (root,P,Path(cfg['kit']),Path(cfg['mesh_site']))]
    project=Path(cfg['root']).resolve()
    def audit(kind,aa):
        if kind in ('socket.connect','socket.bind','subprocess.Popen','os.system'):raise RuntimeError('worker external action denied: '+kind)
        if kind=='open' and isinstance(aa[0],(str,bytes)):
            p=str(Path(os.fsdecode(aa[0])).absolute())
            if p.startswith(str(project)+'/') and p not in allowed and not any(p.startswith(q) for q in prefixes):raise RuntimeError('non-allowlisted project input: '+p)
    sys.addaudithook(audit)
    status='failed';details={};started=time.monotonic()
    with Owner(er/'owners'/args.stage,args.stage,args.episode) as owner:
        try:
            ledger.assert_no_unknown()
            if args.stage=='C':
                from r11_pack import stage_c
                details=stage_c(root,er,ep,cfg,man,ledger,writer,owner)
            else:details={'A':stage_a,'B':stage_b}[args.stage](root,er,ep,cfg,man,ledger,writer,owner)
            ledger.assert_no_unknown();status='completed'
        except BaseException as e:
            status='interrupted' if isinstance(e,BudgetStop) else 'failed';details={'error':repr(e),'traceback':traceback.format_exc()};raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            receipt={'schema':'r11_stage_receipt_v1','episode':args.episode,'stage':args.stage,'attempt':owner.attempt,'producer':man['code']['commit'],'status':status,'finished_at':now(),'elapsed_s':time.monotonic()-started,'details':details}
            path=er/'receipts'/f'{args.stage}_{owner.attempt}.json';atomic_json(path,receipt,True);atomic_json(er/f'{args.stage}_latest.json',ref(path))
if __name__=='__main__':main()
