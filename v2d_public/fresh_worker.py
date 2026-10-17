"""Fresh-only child entry. Preserved numerical functions; no production resume.

Model work requires an official fresh root and a live parent-controller lease.
The fixture path never imports cv2/av/onnxruntime/torch or calls a prediction stage.
"""
from pathlib import Path
import argparse, json, os, signal, sys, time
import numpy as np
import psutil
import r11_state as st
from r12_supervision import control_scope, error
from .common import load, ref, require, sha, digest, now
from .fresh import SOURCE, FreshLedger, verify_fresh_root, runtime_config, validate_budget

def validate_lease(root,man,cfg,episode,stage,path,expected_hash):
    require(sha(path)==expected_hash,'fresh worker lease changed')
    lease=load(path)
    require(lease['schema']=='r12_public_fresh_worker_lease_v1' and lease['root']==str(Path(root).resolve()) and lease['scope']==man['scope'],'fresh lease/root/scope mismatch')
    require(lease['code']==man['code'] and lease['session_ref']==cfg['fresh_session_ref'] and lease['episode']==episode and lease['stage']==stage,'fresh lease/code/session/target mismatch')
    require(os.getppid()==lease['controller_pid'],'worker parent is not the current controller')
    parent=psutil.Process(lease['controller_pid'])
    require(parent.uids().real==os.getuid() and abs(parent.create_time()-lease['controller_create_time'])<.001,'controller identity changed')
    owner=load(Path(root)/'controller_owner/owner_latest.json')
    require(owner['attempt']==lease['controller_attempt'] and owner['pid']==lease['controller_pid'] and not (Path(root)/'controller_owner/attempts'/f'{owner["attempt"]}_end.json').exists(),'controller owner lease no longer active')
    if stage=='fixture':require(man['scope']=='fixture' and lease['model_permission'] is False,'fixture model permission forbidden')
    else:require(man['scope']=='official' and lease['model_permission'] is True,'actual model work requires explicit official controller lease')
    return lease

def dense_stage(root,er,ep,cfg,man,ledger,writer,owner):
    """Original r11_pack.stage_c dense/OBJ core; terminal L1 writer omitted.

    Extraction changes orchestration only. Same float64 dense allocations,
    original-frame indexing, stored identity dtypes, mask, scalar and .17g OBJ.
    Formal Parquet assembly occurs later through the unchanged stock CLI.
    """
    from r11_pack import check_dense,render_audit
    ledger.assert_no_unknown()
    bref=load(er/'B_latest.json');st.verify_ref(bref);b=load(bref['path']);require(b['status']=='completed','B not complete');planref=b['details']['plan'];st.verify_ref(planref);plan=load(planref['path']);br=Path(planref['path']).parent;ids=ep['required']
    identity=st.read_npz(st.verify_ref(b['details']['identity']));mesh=st.read_npz(st.verify_ref(b['details']['mesh']));T=max(ids)+1
    data={'pose':np.full((T,136),np.nan),'scales':identity['scales'],'shape':identity['shape'],'object_rotation':np.full((T,3,3),np.nan),'object_translation':np.full((T,3),np.nan),'object_scale':np.array(1.)};mask=np.zeros(T,bool)
    for f in ids:
        rr=load(br/'frames'/f'{f:06d}.json');require(rr['plan_hash']==digest(plan),'B plan changed');v=st.read_npz(st.verify_ref(rr['output']));data['pose'][f]=v['adapted_params'][:136];data['object_rotation'][f]=v['R'];data['object_translation'][f]=v['t'];mask[f]=True
    check_dense(data,ids,mask);nr=writer.npz(root/'episodes'/f'episode_{ep["episode"]:06d}.npz','C',**data);mr=writer.npz(root/'masks'/f'episode_{ep["episode"]:06d}.npz','C',mask=mask,original_frames=np.array(ids))
    obj=root/'episodes'/f'episode_{ep["episode"]:06d}_object.obj';text='# Input-driven box, local metres; surface quality unknown\n'+''.join('v '+' '.join(format(float(x),'.17g') for x in v)+'\n' for v in mesh['vertices'])+''.join('f '+' '.join(str(int(x)+1) for x in f)+'\n' for f in mesh['faces'])
    require(not any(obj.with_suffix(s).exists() for s in ('.glb','.ply','.stl','.off')),'multiple supported meshes');orr=writer.bytes(obj,text.encode(),'C')
    inputs={'npz':nr,'obj':orr,'mask':mr,'mesh':b['details']['mesh'],'B_receipt':bref,'plan':planref,'episode_input_sha256':digest(ep)}
    writer.json(er/'C_numeric_committed.json',{'schema':'r12_public_fresh_C_commit_v1','frames':ids,'inputs':inputs,'official_stock_writer_executed':False},'C')
    render_audit(er,ep,man,br,writer)
    ledger.assert_no_unknown()
    return {'status':'completed','frames':ids,'inputs':inputs,'quality_audit':ref(er/'quality_audit.json'),'official_stock_writer_executed':False}

def thread_observation(stage):
    value={'opencv_requested':1,'opencv_effective_single_thread':'not_proven','stage':stage}
    if 'cv2' in sys.modules:
        cv=sys.modules['cv2'];value.update(opencv_reported_threads=cv.getNumThreads(),opencv_opencl=cv.ocl.useOpenCL(),opencv_build_sha256=digest(cv.getBuildInformation()))
    else:value['opencv_reported_threads']=None
    if 'torch' in sys.modules:
        torch=sys.modules['torch'];value.update(torch_intra_reported=torch.get_num_threads(),torch_inter_reported=torch.get_num_interop_threads())
    value['ORT']='original stageA requires CPUExecutionProvider and sets sequential intra/inter1; no extra Session is created for this observation'
    return value

def main(argv=None):
    p=argparse.ArgumentParser(description='Private child contract of explicit public fresh controller; never production resume.')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--stage',choices=['A','B','C','fixture'],required=True);p.add_argument('--lease',type=Path,required=True);p.add_argument('--lease-hash',required=True)
    a=p.parse_args(argv);root=a.root.resolve();man=verify_fresh_root(root);cfg,session=runtime_config(root,man)
    validate_lease(root,man,cfg,a.episode,a.stage,a.lease,a.lease_hash)
    ep=load(root/'inputs'/f'episode_{a.episode:06d}.json');ledger=FreshLedger(root,cfg);writer=st.Writer(ledger);er=root/'episode_runs'/f'episode_{a.episode:06d}'
    if a.stage!='fixture':
        from r11_inputs import verify_episode
        verify_episode(ep,cfg)
        for r in cfg['assets'].values():st.verify_ref(r)
    def stop(*args):raise st.BudgetStop('fresh_worker_stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop)
    seconds=min(cfg['stage_limits']['C' if a.stage=='fixture' else a.stage]['wall_s'],session['monotonic_deadlines']['candidate_stop']-time.monotonic())
    require(seconds>0,'fresh worker has no candidate headroom');signal.setitimer(signal.ITIMER_REAL,seconds)
    primary=None;receipt_errors=[];details={};status='failed';started=time.monotonic()
    with control_scope(st.Owner(er/'owners'/a.stage,a.stage,a.episode),er/'control'/('worker_'+uuid_token()),metadata={'mode':'fresh_reproduction','scope':man['scope'],'stage':a.stage,'episode':a.episode}) as owner:
        try:
            validate_budget(root,cfg)
            if a.stage=='fixture':details={'fixture':True,'real_model_calls':0,'real_video_decodes':0}
            elif a.stage=='C':details=dense_stage(root,er,ep,cfg,man,ledger,writer,owner)
            else:
                from r11_worker import stage_a,stage_b
                details={'A':stage_a,'B':stage_b}[a.stage](root,er,ep,cfg,man,ledger,writer,owner)
            validate_budget(root,cfg);status='completed'
        except BaseException as exc:primary=exc
        finally:
            try:signal.setitimer(signal.ITIMER_REAL,0)
            except BaseException as exc:receipt_errors.append(error(exc,'fresh_worker_timer_restore'))
            try:threads=thread_observation(a.stage)
            except BaseException as exc:
                threads={'status':'unknown'};receipt_errors.append(error(exc,'fresh_thread_observation'))
            value={'schema':'r12_public_fresh_stage_v1','scope':man['scope'],'episode':a.episode,'stage':a.stage,'attempt':owner.attempt,
                'producer':man['code']['commit'],'recipe_commit':man['recipe_by_episode'][str(a.episode)],'status':status if not receipt_errors else 'failed',
                'details':details,'finished_at':now(),'elapsed_s':time.monotonic()-started,'primary_error':error(primary,'fresh_stage') if primary else None,
                'receipt_errors':receipt_errors,'thread_observation':threads,'session_ref':cfg['fresh_session_ref']}
            try:
                target=er/'receipts'/f'{a.stage}_{owner.attempt}.json';st.atomic_json(target,value,True);st.atomic_json(er/f'{a.stage}_latest.json',st.ref(target))
            except BaseException as exc:
                receipt_errors.append(error(exc,'fresh_stage_receipt'))
                try:sys.stderr.write(json.dumps(value,ensure_ascii=False)+'\n');sys.stderr.flush()
                except BaseException as exc:receipt_errors.append(error(exc,'fresh_stage_stderr_fallback'))
        if primary is not None:raise primary
        require(not receipt_errors,'fresh worker timer/receipt publication failed')
    return 0

def uuid_token():
    import uuid
    return uuid.uuid4().hex
