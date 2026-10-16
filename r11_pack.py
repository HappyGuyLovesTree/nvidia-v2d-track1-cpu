"""Dense official-kit inputs, exact local sample, per-episode independent roundtrip."""
from pathlib import Path
import argparse, json, sys
import numpy as np
from common import load,sha
from tracking import digest
from r11_state import *

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

def choose_sample(sample,coverage):
    parts=sample.row_id.str.split('_',expand=True);require(parts.shape[1]==5 and (parts[0]=='t1').all(),'sample keys')
    e=parts[1].astype(int);f=parts[2].astype(int);mask=np.zeros(len(sample),bool)
    for ep,ids in coverage.items():mask|=((e==ep)&((f==999999)|f.isin(ids))).to_numpy()
    local=sample.loc[mask,['row_id']].reset_index(drop=True);require(local.row_id.is_unique,'duplicate sample keys')
    return local

def roundtrip(result,coverage,episode_inputs,cfg):
    from v2dlb import mhr_submission as ms
    from v2dlb.mesh_common import _mesh_faces
    rows=[];ps=result.row_id.str.split('_',expand=True)
    for ep,ids in coverage.items():
        take=ps[1].astype(int)==ep;block=result.loc[take,['x','y','z']].copy();p=ps.loc[take]
        block['episode_index']=p[1].astype(int);block['frame_index']=p[2].astype(int);block['array_name']=p[3].astype(int).map(dict(enumerate(ms.array_order())));block['point_index']=p[4].astype(int)
        arrays=ms._to_arrays(block,['x','y','z']);d=read_npz(episode_inputs[ep]['npz']['path']);mesh=read_npz(episode_inputs[ep]['mesh']['path'])
        poses=ms._unpack(arrays['mhr_pose'],136,'mhr_pose');scales=ms._unpack(arrays['mhr_scales'],68,'mhr_scales');shape=ms._unpack(arrays['mhr_shape'],45,'mhr_shape')
        require(np.array_equal(poses,d['pose'][ids]) and np.array_equal(scales,d['scales']) and np.array_equal(shape,d['shape']),'episode pose/identity roundtrip')
        projected=ms._rotations(arrays['object_rotation']);scale=ms._object_scale(arrays['object_scale'])
        require(np.array_equal(arrays['object_rotation'],d['object_rotation'][ids]) and np.array_equal(arrays['object_translation'][:,0],d['object_translation'][ids]) and scale==float(d['object_scale']),'R row/translation/scale roundtrip')
        vv=arrays['object_mesh_vertices'];ff=_mesh_faces(arrays['object_mesh_faces'],len(vv));tri=vv[ff];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        # Budget is read from this sample episode, then checked against the frozen kit contract.
        require(vv.shape==(4096,3) and ff.shape==(4096,3) and np.isfinite(areas).all() and areas.sum()>0 and np.count_nonzero(areas)==12,'mesh budget/area')
        require(np.array_equal(vv[8:],np.repeat(vv[:1],len(vv)-8,axis=0)) and np.array_equal(ff[12:],np.zeros((len(ff)-12,3),int)),'mesh padding')
        require(np.array_equal(np.unique(vv,axis=0),np.unique(mesh['vertices'],axis=0)),'input-driven mesh roundtrip')
        require(np.max(np.abs(d['object_rotation'][ids].transpose(0,2,1)@d['object_rotation'][ids]-np.eye(3)))<=1e-6,'R orthogonality')
        for i,fid in enumerate(ids):
            direct=vv@d['object_rotation'][fid].T+d['object_translation'][fid]
            official=(arrays['object_rotation'][i]@vv.T).T*scale+arrays['object_translation'][i,0]
            require(np.abs(direct-official).max()<=1e-10,'object transform roundtrip')
        rows.append({'episode':ep,'frames':ids,'rows':int(take.sum()),'pose_identity_tail_R_translation_mesh':'passed','mesh_area_m2':float(areas.sum()),'R_projection_max_change':float(np.abs(projected-d['object_rotation'][ids]).max())})
    return rows

def pack_local(root,cfg,man,coverage,episode_inputs,writer,label):
    import pandas as pd
    sys.path.insert(0,cfg['mesh_site']);sys.path.insert(0,cfg['kit'])
    from tools import pack_reconstruction as pack
    original={str(p):sha(p) for p in Path(cfg['kit']).rglob('*.py')};sample=pd.read_parquet(cfg['assets']['sample']['path'],columns=['row_id'])
    directory=root/'packs'/label
    if (directory/'validation.json').exists():
        existing=load(directory/'validation.json');require(existing['inputs']=={str(e):v for e,v in episode_inputs.items()} and existing['completed']=={str(e):ids for e,ids in coverage.items()} and existing['producer']==man['code']['commit'],'existing immutable pack parent conflict')
        verify_ref(existing['candidate']);verify_ref(existing['sample'])
        return existing
    local=choose_sample(sample,coverage);expected=sum(50*len(ids)+8231 for ids in coverage.values());require(len(local)==expected,'actual coverage/sample row count mismatch')
    total=sum(map(len,coverage.values()));full=len(coverage)==30 and total==9877
    if full:require(local.row_id.tolist()==sample.row_id.tolist() and len(local)==740780,'full sample exact coverage')
    directory=root/'packs'/label;directory.mkdir(parents=True,exist_ok=True);samplepath=directory/'sample.parquet'
    outbuf=io.BytesIO();local.to_parquet(outbuf,index=False);writer.bytes(samplepath,outbuf.getvalue(),'C')
    marker='local-git:'+man['code']['commit'];require(not bool(__import__('re').match(pack.rs.CODE_COMMIT_PATTERN,marker)),'local marker unexpectedly official')
    outfile=directory/'candidate.parquet'
    def write_local(rows,sample,commit,out):
        require(commit==marker and __import__('re').fullmatch(r'local-git:[0-9a-f]{40}',commit),'incorrect local code marker')
        require(rows.row_id.is_unique and sample.row_id.is_unique and len(rows)==len(sample),'row keys duplicate/count mismatch')
        merged=sample.merge(rows,on='row_id',how='left',sort=False,validate='one_to_one');require(merged.row_id.tolist()==sample.row_id.tolist() and np.isfinite(merged[['x','y','z']].to_numpy()).all(),'exact key/finite writer check')
        merged['code_commit_url']=commit;b=io.BytesIO();merged[['row_id','x','y','z','code_commit_url']].to_parquet(b,index=False);writer.bytes(out,b.getvalue(),'C')
    stock=pack.write;pack.write=write_local
    try:pack.pack_track1(argparse.Namespace(episodes=root/'episodes',sample=samplepath,commit=marker,out=outfile))
    finally:pack.write=stock
    result=pd.read_parquet(outfile);require(result.columns.tolist()==['row_id','x','y','z','code_commit_url'] and result.row_id.tolist()==local.row_id.tolist() and result.row_id.is_unique and result.code_commit_url.unique().tolist()==[marker] and np.isfinite(result[['x','y','z']].to_numpy()).all(),'candidate readback')
    checked=roundtrip(result,coverage,episode_inputs,cfg);require(all(sha(p)==h for p,h in original.items()),'official kit modified')
    validation={'schema':'r11_pack_receipt_v1','producer':man['code']['commit'],'coverage':'full' if full else 'partial','episode_count':len(coverage),'frame_count':total,'rows':len(result),'completed':{str(e):ids for e,ids in coverage.items()},'per_episode':checked,'numeric_index_mesh_pack':'passed','sample':ref(samplepath),'candidate':ref(outfile),'inputs':{str(e):v for e,v in episode_inputs.items()},'provenance_kind':'local_git','submission_eligible':False,'official_writer':'not_used_original_terminal_write_local_L1','official_code_commit_url_format':'failed_local_marker_expected','remote_commit_accessibility':'unpublished','kaggle_scoring':'not_run','public_row':'not_run','quality':'contact_real_scale_surface_unknown','as_of':now()}
    writer.json(directory/'validation.json',validation,'C');writer.json(directory/'candidate.sidecar.json',validation,'C')
    return validation

def render_audit(er,ep,man,br,writer):
    import cv2
    cv2.setNumThreads(1);statuses=[]
    from r11_worker import SCORED
    for e,f in man['audit']['pairs']:
        if e!=ep['episode']:continue
        path=er/'audit'/f'{f:06d}_source.jpg'
        if not path.exists():statuses.append({'frame':f,'status':'pending','reason':'audit source image unavailable'});continue
        try:
            record=load(er/'frames'/f'{f:06d}.json');r=load(br/'frames'/f'{f:06d}.json');a=read_npz(verify_ref(r['output']));canvas=cv2.imread(str(path));require(canvas is not None,'source image unreadable');K=np.asarray(record['transform']['K'])
            def project(x):return x[:,:2]/x[:,2:3]*K[0,0]+K[:2,2]
            for points,color,rad in ((a['head_camera'],(80,255,80),3),(a['adapter_joints'][SCORED],(255,160,40),5)):
                for pt in project(points):
                    if np.isfinite(pt).all() and np.abs(pt).max()<1e6:cv2.circle(canvas,tuple(np.rint(pt).astype(int)),rad,color,-1)
            x1,y1,x2,y2=map(int,record['decision']['bbox']);cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,200,255),3)
            uv=project(a['vertices_camera'])
            for i,j in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):
                if np.isfinite(uv[[i,j]]).all() and np.abs(uv[[i,j]]).max()<1e6:cv2.line(canvas,tuple(np.rint(uv[i]).astype(int)),tuple(np.rint(uv[j]).astype(int)),(255,80,230),2)
            cv2.putText(canvas,f'episode {e} original {f} | {record["decision"]["branch"]} | quality UNKNOWN',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            ok,encoded=cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,94]);require(ok,'overlay encode failed');rr=writer.bytes(er/'audit'/f'{f:06d}_overlay.jpg',encoded.tobytes(),'images');statuses.append({'frame':f,'status':'generated_not_human_reviewed','overlay':rr,'rgb_sha256':record['input_identity']['rgb_sha256'],'image_encoding':'lossy_jpeg94_for_review_only'})
        except Exception as exc:statuses.append({'frame':f,'status':'unavailable','error':repr(exc)})
    writer.json(er/'quality_audit.json',{'as_of':now(),'frames':statuses,'quality':'unknown','virtual_K':'not measured calibration'})
    return statuses

def stage_c(root,er,ep,cfg,man,ledger,writer,owner):
    bref=load(er/'B_latest.json');verify_ref(bref);b=load(bref['path']);require(b['status']=='completed','B not complete');planref=b['details']['plan'];verify_ref(planref);plan=load(planref['path']);br=Path(planref['path']).parent;ids=ep['required']
    identity=read_npz(verify_ref(b['details']['identity']));mesh=read_npz(verify_ref(b['details']['mesh']));T=max(ids)+1
    data={'pose':np.full((T,136),np.nan),'scales':identity['scales'],'shape':identity['shape'],'object_rotation':np.full((T,3,3),np.nan),'object_translation':np.full((T,3),np.nan),'object_scale':np.array(1.)};mask=np.zeros(T,bool)
    for f in ids:
        rr=load(br/'frames'/f'{f:06d}.json');require(rr['plan_hash']==digest(plan),'B plan changed');v=read_npz(verify_ref(rr['output']));data['pose'][f]=v['adapted_params'][:136];data['object_rotation'][f]=v['R'];data['object_translation'][f]=v['t'];mask[f]=True
    check_dense(data,ids,mask);nr=writer.npz(root/'episodes'/f'episode_{ep["episode"]:06d}.npz','C',**data);mr=writer.npz(root/'masks'/f'episode_{ep["episode"]:06d}.npz','C',mask=mask,original_frames=np.array(ids))
    obj=root/'episodes'/f'episode_{ep["episode"]:06d}_object.obj';text='# Input-driven box, local metres; surface quality unknown\n'+''.join('v '+' '.join(format(float(x),'.17g') for x in v)+'\n' for v in mesh['vertices'])+''.join('f '+' '.join(str(int(x)+1) for x in f)+'\n' for f in mesh['faces'])
    require(not any(obj.with_suffix(s).exists() for s in ('.glb','.ply','.stl','.off')),'multiple supported meshes');orr=writer.bytes(obj,text.encode(),'C')
    inputs={ep['episode']:{'npz':nr,'obj':orr,'mask':mr,'mesh':b['details']['mesh'],'B_receipt':bref,'plan':planref,'episode_input_sha256':digest(ep)}}
    result=pack_local(root,cfg,man,{ep['episode']:ids},inputs,writer,f'episode_{ep["episode"]:06d}')
    writer.json(er/'C_numeric_committed.json',{'schema':'r11_C_numeric_commit_v1','pack':ref(root/'packs'/f'episode_{ep["episode"]:06d}'/'validation.json'),'frames':ids,'inputs':inputs[ep['episode']],'quality_audit':None},'C')
    result['quality_audit']=render_audit(er,ep,man,br,writer)
    return {'status':'completed','pack':ref(root/'packs'/f'episode_{ep["episode"]:06d}'/'validation.json'),'frames':ids,'inputs':inputs[ep['episode']],'quality_audit':ref(er/'quality_audit.json')}
