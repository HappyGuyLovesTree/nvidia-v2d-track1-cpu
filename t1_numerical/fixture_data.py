"""Build original-format, wholly synthetic R0 inputs; no real asset/data access."""
import io,json,hashlib
from pathlib import Path
import numpy as np
from control import reference,write_new,compact_digest,legacy_array_hash,legacy_arrays_hash
from portable import save_npz

def build(base):
    base=Path(base).resolve();base.mkdir();run=base/'r0';run.mkdir();(run/'inputs').mkdir();(run/'budget').mkdir()
    for name in ('episodes','masks','meshes','episode_runs'):(run/name).mkdir()
    log=run/'budget/calls.jsonl'
    def end_pair(model,e,f,stage,inp,out):
        cid=f'fixture_{model}_{e}_{f}_{stage}'
        start=dict(kind='call_start',call_id=cid,model_kind=model,episode=e,frame=f,stage=stage,attempt='fixture',ordinal=f+1,input_identity=inp)
        end=dict(start,kind='call_end',outcome='returned',output_identity=out)
        with log.open('ab') as file:
            file.write((json.dumps(start,sort_keys=True,separators=(',',':'))+'\n').encode())
            offset=file.tell();raw=(json.dumps(end,sort_keys=True,separators=(',',':'))+'\n').encode();file.write(raw)
        return dict(path=str(log),offset=offset,length=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    specs=[]
    for e,N in ((0,30),(1,8)):
        er=run/'episode_runs'/f'episode_{e:06d}';er.mkdir();(er/'frames').mkdir();(er/'hmr').mkdir()
        ids=list(range(N));ep=dict(schema='r11_episode_input_v1',episode=e,required=ids,sources=[ids],source_map={str(f):0 for f in ids},
            video=dict(path='not-a-real-fixture-video.mp4',sha256='a'*64),whole_pts=dict(path='fixture-pts-not-read.json',sha256='b'*64),
            whole_pts_verified=True,local_file_verified=True,stream=dict(width=1000,height=1000,time_base='1/30'))
        write_new(run/'inputs'/f'episode_{e:06d}.json',ep);eh=compact_digest(ep)
        records={};raws={};previous=None;journal=[];previous_commit=None;sh=np.zeros(45,np.float32);sc=np.ones(68,np.float32)
        zs=np.array([5.+[-.03,.01,-.01,0.,0.,.02,-.02][i%7] for i in ids],np.float32)
        if e==0:zs[7]=5.08;zs[18]=8.
        for f in ids:
            p=np.zeros((1,204),np.float32);p[0,:3]=[.3,0.,.3];p[0,3:136]=np.arange(133)*.001;p[0,136:]=sc
            raw=dict(mhr_params=p,shape_params=sh[None],cam_trans=np.array([[0.,0.,zs[f]]],np.float32),joints_2d=np.zeros((1,70,2),np.float32),joints_3d=np.zeros((1,70,3),np.float32))
            rawref=save_npz(er/'hmr'/f'{f:06d}.npz',raw);raws[f]=raw
            tensors=dict(image='c'*64,cliff_cond='d'*64)
            href=end_pair('hmr',e,f,'A',compact_digest(tensors),legacy_arrays_hash(raw))
            candidates=[dict(index=0,class_id=0,xyxy=[200.,200.,300.,500.],score=.9,tier='high',valid=True)]
            if e==1:candidates.append(dict(index=1,class_id=0,xyxy=[400.,200.,500.,500.],score=.8,tier='high',valid=True))
            r=dict(schema='r11_frame_v1',status='frame_committed',episode=e,frame=f,source=0,producer='1'*40,numeric_hash='2'*64,runtime_sha256='3'*64,
                episode_input_sha256=eh,previous_frame_receipt=previous,raw=rawref,hmr_outcome='hmr_valid',hmr_call_end=href,
                transform=dict(K=[[1000.,0.,500.],[0.,1000.,500.],[0.,0.,1.]]),
                input_identity=dict(episode=e,source=0,frame=f,pts=f,time_base='1/30',width=1000,height=1000,video_sha256='a'*64,whole_pts_sha256='b'*64,hmr_tensor_hashes=tensors),
                decision=dict(episode=e,frame_index=f,source_segment=0,source_discontinuity=False,track_segment=0,method='single_iou_low_v1',branch='associated_detection',
                    bbox=[200.,200.,300.,500.],selected_index=0,selected_tier='high',lost_count=0,reason='fixture',input_parents=dict(RGB='4'*64,wholePTS='b'*64),candidates=candidates))
            ref=write_new(er/'frames'/f'{f:06d}.json',r);records[f]=r;previous=compact_digest(r)
            item=dict(frame=f,record=ref,previous_commit=previous_commit,committed_at='fixture-not-a-real-timestamp');journal.append(item);previous_commit=compact_digest(item)
        write_new(er/'frame_commits.json',journal)
        bp=dict(schema='r11_B_plan_v1',producer='1'*40,episode_input=eh,frames={str(f):compact_digest(records[f]) for f in ids},shape=sh.tolist(),scales=sc.tolist(),
                parents={str(f):dict(frame=f,raw_sha256=records[f]['raw']['sha256'],reason='original_valid') for f in ids})
        ph=compact_digest(bp);br=er/'B'/ph;br.mkdir(parents=True);(br/'frames').mkdir();bpref=write_new(br/'plan.json',bp)
        write_new(er/'B_plan_latest.json',dict(path=bpref['path'],sha256=bpref['sha256'],plan_hash=ph))
        pose=np.full((N+3,136),np.nan);R=np.full((N+3,3,3),np.nan);t=np.full((N+3,3),np.nan);mask=np.zeros(N+3,bool)
        for f in ids:
            original=raws[f]['mhr_params'][0].copy();q=original.copy();q[:3]+=np.asarray(10*np.array([1,-1,-1])*raws[f]['cam_trans'][0].astype(float),np.float32)
            pose[f]=q[:136];R[f]=np.eye(3);t[f]=raws[f]['cam_trans'][0];t[f,0]=-0.;mask[f]=True
            output=save_npz(br/'frames'/f'{f:06d}.npz',dict(adapted_params=q,R=R[f].copy(),t=t[f].copy()))
            ends=[]
            for label,p in (('baseline',original),('adapted',q)):
                callinput=compact_digest(dict(params=legacy_array_hash(p),shape=legacy_array_hash(sh[None]),plan=ph))
                ends.append(end_pair('rig',e,f,'B:'+label,callinput,'e'*64))
            write_new(br/'frames'/f'{f:06d}.json',dict(frame=f,source=0,plan_hash=ph,hmr_fill=bp['parents'][str(f)],output=output,call_ends=ends))
        # Deliberate non-default NaN payload in unselected slots for bit tests.
        pose.view(np.uint64)[N+1,17]=0x7ff8000000000123;t.view(np.uint64)[N+2,0]=0x7ff8000000000456
        dense=save_npz(run/'episodes'/f'episode_{e:06d}.npz',dict(pose=pose,scales=sc,shape=sh,object_rotation=R,object_translation=t,object_scale=np.array(1.)))
        mr=save_npz(run/'masks'/f'episode_{e:06d}.npz',dict(mask=mask,original_frames=np.array(ids)))
        vertices=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]])
        faces=np.array([[0,1,2]],np.int64);mesh=save_npz(run/'meshes'/f'episode_{e:06d}.npz',dict(vertices=vertices,faces=faces))
        obj=write_new(run/'episodes'/f'episode_{e:06d}_object.obj',b'v 0 0 0\nv 0.10000000000000001 0 0\nv 0 0.10000000000000001 0\nf 1 2 3\n')
        specs.append(dict(episode=e,run_root='r0',dense=str(Path(dense['path']).relative_to(base)),mask=str(Path(mr['path']).relative_to(base)),obj=str(Path(obj['path']).relative_to(base)),mesh=str(Path(mesh['path']).relative_to(base))))
    config=dict(schema='t1_r0_dataset_v1',fixture=True,episodes=specs,prefix_map=[],expected=dict(episodes=2,required_frames=38,source_runs=2,allowed_segments=1,allowed_frames=30,proposals=2,geometric_accepts=1))
    write_new(base/'dataset.json',config);return base/'dataset.json'

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out');args=p.parse_args();print(build(args.out))
