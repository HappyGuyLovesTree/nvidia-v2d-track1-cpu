"""R0 -> original P scope -> fixed T1 -> geometry -> numerical six-array export.

Public numerical reproduction only. No human quality receipt or submission API.
"""
import argparse,collections,hashlib,io,json,os,shutil,sys
from pathlib import Path
import numpy as np
import planner
import t1
from geometry import need,ContractError
from geometry_d import adapt_z,params204,all_gates,SCORED
from d_common import camera,identity,output_identity,MODEL_SHA,bit_equal,GEOMETRY_POLICY
from control import reference,verify,write_new,digest,legacy_array_hash
from r0_io import Dataset,DENSE
from runtime import Calls,supervise,RESOURCE_POLICY

SOURCE=Path(__file__).resolve().parent
def save_npz(path,arrays):
    b=io.BytesIO();np.savez_compressed(b,**arrays);return write_new(path,b.getvalue())
def source_tree():return digest({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCE.glob('*.py'))})
def expected(config,key,value):
    if key in config.get('expected',{}):need(config['expected'][key]==value,'caller expected-count contract: '+key)

def statistical_plan(dataset,out):
    controls=dataset.controls();P=planner.plan(controls);segments=[s for s in P['segments'] if s['action']=='update']
    context=dataset.contexts(segments);decisions=[];proposals=[]
    for segment in segments:
        e=segment['key'][0];rows=[context[e,f] for f in segment['frames']]
        cams={r['frame']:tuple(float(x) for x in r['cam']) if r['current'] else None for r in rows}
        for r,d in zip(rows,t1.audit_segment(rows,cams)):
            d.update(episode=e,source=r['source'],segment_id=r['segment_id'])
            if d['proposed']:
                d['proposal_id']=digest(dict(episode=e,source=r['source'],frame=r['frame'],target_delta_z_m=d['delta_z_m']))
                d['target_delta_z_m']=d['delta_z_m']
                proposals.append(d)
            decisions.append(d)
    counts=dict(episodes=len(dataset.ep),required_frames=len(controls),source_runs=P['source_count'],
                allowed_segments=len(segments),allowed_frames=P['update_N'],proposals=len(proposals))
    for key,value in counts.items():expected(dataset.config,key,value)
    pref=write_new(out/'P_scope.json',P)
    rawref=write_new(out/'raw_statistical_proposals.json',dict(schema='t1_recomputed_raw_proposals_v1',rows=proposals,original_R0_only=True))
    sref=write_new(out/'statistical_proposals.json',dict(schema='t1_proposal_index_v1',source_ref=rawref,
        binding=dict(source_tree_sha256=source_tree(),policy_sha256=digest(t1.POLICY),producer='portable_statistical_replay'),rows=proposals))
    write_new(out/'statistical_decisions.json',dict(rows=decisions,counts=counts,first_reason_counts=dict(collections.Counter(d['reason'] for d in decisions)),
        outside_P_inherited=len(controls)-P['update_N'],P_scope=pref))
    return P,proposals,counts,sref

def execute(dataset_path,out,mode,fixture,model_path,rig_cap,token):
    origin=json.loads((out/'origin.json').read_text())
    need(origin['supervisor']['pid']==os.getppid() and origin['authority']['token']==token,'live portable supervisor required')
    dataset=Dataset(dataset_path);need(bool(dataset.config.get('fixture',False))==fixture,'fixture/data scope conflict')
    P,proposals,counts,index=statistical_plan(dataset,out)
    base=dict(schema='t1_portable_numerical_result_v1',fixture=fixture,source_tree_sha256=source_tree(),counts=counts,
              baseline_producers=sorted({r['record']['producer'] for r in dataset.records.values()}),statistical_policy=t1.POLICY,
              proposal_index=index,quality_pass=False,submission_eligible=False,human_quality_receipts=None,
              HMR=0,detector=0,video_decode=0)
    if mode=='plan' or not proposals:
        write_new(out/'input_read_manifest.json',dict(entries=list(dataset.entries.values()),call_parent_validation='selected start/end closure; whole-raw output identities opaque'))
        write_new(out/'numerical_result.json',dict(base,status='statistical_plan_only' if mode=='plan' else 'no_proposals_stop',real_model_loads=0,real_rig=0))
        return
    centers={(p['episode'],p['frame']):p for p in proposals};rows=set();edges=set()
    for e,f in centers:
        sid=dataset.context[e,f]['segment_id']
        for n in (f-1,f,f+1):
            need((e,n) in dataset.context and dataset.context[e,n]['segment_id']==sid,'prospective neighbor not same P segment')
            rows.add((e,n))
        edges.update(((e,f-1,f),(e,f,f+1)))
    required_calls=len(rows)+len(centers);need(rig_cap>=required_calls,'rig cap below derived all-proposal scope')
    for e in dataset.ep:dataset.load_episode(e)
    selected={k:dataset.geometry(*k) for k in sorted(rows)}
    inputs_ref=write_new(out/'input_read_manifest.json',dict(entries=list(dataset.entries.values()),fixture=fixture,
        selected_rows=[list(k) for k in sorted(rows)],proposals=len(centers),reference_rows=len(rows),edges=len(edges),
        original_parent_scope='closed HMR and B metadata; actual adapted rig input q/shape/scales rederived; other raw heads not decoded'))
    calls=Calls(out,dict(rig=required_calls,rig_load=1,hmr=0,hmr_load=0));model=None
    modelref=None
    if not fixture:
        need(model_path is not None,'explicit fixed rig path required');modelref=reference(Path(model_path).resolve())
        need(modelref['sha256']==MODEL_SHA,'fixed MHR model hash conflict')
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with calls.call('rig_load','once',dict(fixture=fixture,asset=modelref,input_manifest=inputs_ref)) as call:
        if not fixture:
            model=torch.jit.load(verify(modelref),map_location='cpu').eval()
            need([a.name for a in list(model.forward.schema.arguments)[1:]]==['identity_coeffs','model_parameters','face_expr_coeffs','apply_correctives'],'fixed MHR schema')
        info=dict(fixture=fixture,asset=modelref,coordinate='mhr_native',actual_model_loaded=not fixture)
        call['output']=write_new(out/'rig_load.json',info);call['output_identity']=digest(info)
    bindings=[];native={};rigs={};final_rigs={};final={};final_cam={};results=[]
    def forward(params,shape):
        if fixture:
            # Clearly synthetic affine rig. Never advertised as real MHR output.
            v=np.zeros((1,18439,3),np.float32);v[0,:,0]=np.linspace(-20,20,18439);v[0,:,1]=np.linspace(-50,50,18439)
            v[0]+=params[:3]*10
            j=np.zeros((1,127,8),np.float32);j[0,:,:3]=v[0,:127]
            return v,j
        import torch
        with torch.inference_mode():v,j=model(torch.tensor(shape[None],dtype=torch.float32),torch.tensor(params[None],dtype=torch.float32),torch.zeros((1,72),dtype=torch.float32),True)
        return v.cpu().numpy(),j.cpu().numpy()
    def rig(k,values,role):
        p,s=params204(values);name=f'ep{k[0]:06d}_f{k[1]:06d}_{role}'
        parents=dict(fixture=fixture,role=role,q=identity(values['pose']),shape=identity(values['shape']),scales=identity(values['scales']),
                     actual_params204=identity(p),actual_shape45=identity(s),model_sha256=modelref['sha256'] if modelref else 'explicit_synthetic_mock',
                     source_tree_sha256=source_tree(),policy_sha256=digest(GEOMETRY_POLICY),proposal_index_sha256=index['sha256'],input_manifest=inputs_ref)
        with calls.call('rig',name,parents) as call:
            v,j=forward(p,s);raw=save_npz(out/(name+'_raw.npz'),dict(vertices=v,joints=j))
            need(v.shape==(1,18439,3) and j.shape==(1,127,8),'actual rig forward domains')
            values=dict(vertices=v[0].copy(),joints=j[0,:,:3].copy());ref=save_npz(out/(name+'_native.npz'),values)
            call['output']=ref;call['output_identity']=output_identity(values)
        record=dict(call_id=call['call_id'],output=ref,output_identity=call['output_identity'],raw_forward=raw,parents=parents)
        journal=calls.rows();start=next(r for r in journal if r.get('call_id')==call['call_id'] and r['event']=='start');end=next(r for r in journal if r.get('call_id')==call['call_id'] and r['event']=='end')
        record['call_receipt']=write_new(out/(name+'_call.json'),dict(schema='t1_persisted_rig_call_v1',fixture=fixture,coordinate='mhr_native',
            binding=dict(run_id=origin['run_id'],policy_sha256=digest(GEOMETRY_POLICY),source_tree_sha256=source_tree(),model_sha256=parents['model_sha256']),
            start=start,end=dict(end,kind='rig'),raw_forward_ref=raw,camera_transform='camera_m=native*diag(1,-1,-1)/100',
            original_end_kind_field='canonical receipt adds kind=rig; original journal unchanged'))
        bindings.append(record);return values,camera(values),record
    for k,(old,cam,K,v,faces) in selected.items():
        n,c,b=rig(k,old,'R0_reference');need(all(np.isfinite(x).all() for x in c.values()),'baseline rig nonfinite')
        native[k]=n;rigs[k]=c;final_rigs[k]=c;final[k]={n:a.copy() for n,a in old.items()};final_cam[k]=cam.astype(np.float64)
    for k,d in centers.items():
        old,cam,K,v,faces=selected[k];new,adapter=adapt_z(old,d['target_z_m'],cam)
        n,c,b=rig(k,new,'T1_proposal');gate=all_gates(old,new,v,rigs[k],c,K,adapter)
        if gate['accepted']:
            final[k]=new;final_rigs[k]=c;final_cam[k][2]=d['target_z_m']
        else:need(all(bit_equal(old[f],final[k][f]) for f in old),'whole-frame exact R0 fallback')
        def params(a):return dict(q=a['pose'],shape=a['shape'],scales=a['scales'],R=a['object_rotation'],t=a['object_translation'],scale=a['object_scale'],K=K)
        label=f'ep{k[0]:06d}_f{k[1]:06d}'
        oldref=save_npz(out/(label+'_old_params.npz'),params(old));newref=save_npz(out/(label+'_candidate_params.npz'),params(new))
        # Calls with equal numerical parameters can exist; bind by the actual key.
        oldstart=next(x for x in calls.rows() if x['event']=='start' and x['key']==label+'_R0_reference')
        oldbinding=next(x for x in bindings if x['call_id']==oldstart['call_id'])
        results.append(dict(episode=k[0],source=dataset.context[k]['source'],frame=k[1],proposal_id=d['proposal_id'],target_delta_z_m=d['delta_z_m'],
            old_params=oldref,candidate_params=newref,old_rig=oldbinding['output'],new_rig=b['output'],old_call=oldbinding['call_receipt'],new_call=b['call_receipt'],
            decision='accepted' if gate['accepted'] else 'inherit_R0',reason=gate['first_reason'],
            adapter=adapter,gate=gate,original_dense_ref=dataset.pin(dataset.ep[k[0]]['spec']['dense'],DENSE),
            mesh_binding=dict(obj_ref=dataset.pin(dataset.ep[k[0]]['spec']['obj'],['bytes']),mesh_npz_ref=dataset.pin(dataset.ep[k[0]]['spec']['mesh'],['vertices','faces']),
                vertices_identity=identity(v),faces_identity=identity(faces),coordinate='local_metres',old_transform_params=oldref,candidate_transform_params=newref,
                transform='(vertices*scale)@R.T+t; original local metres')))
    edge_results=[]
    for e,a,b in sorted(edges):
        x,y=(e,a),(e,b);ox,oy=selected[x][0],selected[y][0]
        edge_results.append(dict(episode=e,a=a,b=b,cam_old=float(np.linalg.norm(selected[y][1].astype(float)-selected[x][1])),cam_new=float(np.linalg.norm(final_cam[y]-final_cam[x])),
            q_old=float(np.linalg.norm((oy['pose'][:3]-ox['pose'][:3])/10)),q_new=float(np.linalg.norm((final[y]['pose'][:3]-final[x]['pose'][:3])/10)),
            joint22_old=np.linalg.norm(rigs[y]['joints'][SCORED]-rigs[x]['joints'][SCORED],axis=1).tolist(),joint22_new=np.linalg.norm(final_rigs[y]['joints'][SCORED]-final_rigs[x]['joints'][SCORED],axis=1).tolist(),
            human_motion_time_gate='not assessed by public numeric replay'))
    accepted=sum(r['gate']['accepted'] for r in results);expected(dataset.config,'geometric_accepts',accepted)
    decisionref=write_new(out/'geometry_decisions.json',dict(results=results,edges=edge_results,bindings=bindings,quality_pass=False))
    bundle=write_new(out/'geometry_bundle.json',dict(schema='t1_persisted_geometry_v1',fixture=fixture,proposal_index=index,rows=results,
        binding=dict(run_id=origin['run_id'],policy_sha256=digest(GEOMETRY_POLICY),source_tree_sha256=source_tree(),proposal_index_sha256=index['sha256'],
                     model_sha256=modelref['sha256'] if modelref else 'explicit_synthetic_mock'),
        controlled_results=decisionref,all_old_reference_rig_outputs=[b for b in bindings if b['parents']['role']=='R0_reference'],
        coordinate='mhr_native',quality_pass=False,independent_D_acceptance=None))
    exports=[]
    if accepted:
        for e,info in dataset.ep.items():
            spec=info['spec'];old=dataset.load_episode(e)[0];new={k:v.copy() for k,v in old.items()}
            changed={f for ee,f in centers if ee==e and next(r for r in results if (r['episode'],r['frame'])==(ee,f))['gate']['accepted']}
            for f in changed:new['pose'][f,2]=final[e,f]['pose'][2];new['object_translation'][f,2]=final[e,f]['object_translation'][2]
            for key in set(DENSE)-{'pose','object_translation'}:need(bit_equal(old[key],new[key]),'static dense bit invariant')
            for key in ('pose','object_translation'):
                restore=new[key].copy();restore[:,2]=old[key][:,2]
                need(bit_equal(restore,old[key]),'only z slots may change')
                unselected=[i for i in range(len(old[key])) if i not in changed]
                need(bit_equal(new[key][unselected],old[key][unselected]),'unselected frame/NaN layout bit invariant')
            target=out/'episodes';target.mkdir(exist_ok=True);ref=save_npz(target/f'episode_{e:06d}.npz',new)
            copied={}
            for key,suffix in (('mask','mask.npz'),('obj','object.obj'),('mesh','mesh.npz')):
                src=dataset.resolve(spec[key]);dst=target/f'episode_{e:06d}_{suffix}';copied[key]=write_new(dst,src.read_bytes())
                need(copied[key]['sha256']==reference(src)['sha256'],'mask/OBJ/mesh byte-copy invariant')
            exports.append(dict(episode=e,dense=ref,copied=copied,changed_frames=sorted(changed)))
    write_new(out/'numerical_result.json',dict(base,status='numerical_geometry_only' if accepted else 'zero_acceptance_stop_no_export',
        reference_rows=len(rows),affected_edges=len(edges),geometric_accepts=accepted,first_reason_counts=dict(collections.Counter(r['reason'] for r in results)),
        rig_load=1,rig_total=required_calls,real_model_loads=0 if fixture else 1,real_rig=0 if fixture else required_calls,synthetic_rig=required_calls if fixture else 0,
        geometry_bundle=bundle,exports=exports,independent_quality='not performed',stock_pack='not performed'))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['plan','run','_worker'])
    parser.add_argument('--dataset',required=True);parser.add_argument('--out',required=True);parser.add_argument('--rig');parser.add_argument('--rig-cap',type=int,default=0)
    parser.add_argument('--fixture',action='store_true');parser.add_argument('--execute-models',action='store_true');parser.add_argument('--seconds',type=float,default=1200)
    parser.add_argument('--worker-mode');parser.add_argument('--token')
    args=parser.parse_args();out=Path(args.out).resolve()
    need(args.seconds>0 and args.seconds<=1200,'portable seconds cap')
    need(not(args.fixture and args.execute_models),'fixture cannot execute real models')
    if args.mode=='_worker':return execute(Path(args.dataset).resolve(),out,args.worker_mode,args.fixture,args.rig,args.rig_cap,args.token)
    need(args.mode=='plan' or args.fixture or args.execute_models,'run needs explicit fixture or model execution flag')
    need(args.mode=='plan' or args.rig_cap>0,'explicit rig cap required for geometry')
    need(not out.exists() and out!=SOURCE and SOURCE not in out.parents and out not in SOURCE.parents,'new output outside source required');out.mkdir(parents=True)
    import uuid
    token=uuid.uuid4().hex;cmd=[sys.executable,str(Path(__file__).resolve()),'_worker','--dataset',str(Path(args.dataset).resolve()),'--out',str(out),'--worker-mode',args.mode,'--token',token,'--rig-cap',str(args.rig_cap)]
    if args.fixture:cmd.append('--fixture')
    if args.rig:cmd+=['--rig',str(Path(args.rig).resolve())]
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    policy=dict(RESOURCE_POLICY,seconds=args.seconds,hmr=0,hmr_load=0,rig=max(0,args.rig_cap),rig_load=1 if args.mode=='run' else 0)
    receipt=supervise(out,cmd,policy,authority=dict(token=token,source_tree_sha256=source_tree(),mode=args.mode),fixture=args.fixture)
    need(receipt['status']=='completed','portable worker stopped; preserve output, no automatic retry')
    print(json.dumps(dict(status=receipt['status'],result=str(out/'numerical_result.json'),fixture=args.fixture)))

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('portable reproduction stopped:',type(e).__name__,str(e),file=sys.stderr);raise SystemExit(1)
