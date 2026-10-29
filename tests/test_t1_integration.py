"""Synthetic evidence only. No inference, private predictions, network or upload."""
from pathlib import Path
import argparse,copy,json,subprocess,sys,unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v2d_t1.core import *
from v2d_t1.lineage import *
from v2d_t1.package import full_invariants,load_contract,assemble,roundtrip,mesh_binding,mesh_arrays,ORIGINAL32
from v2d_t1 import gate

ROOT=Path(__file__).resolve().parents[1]

def fixture(out,kit,stock_python,dz=.001):
    out=Path(out);out.mkdir(parents=True,exist_ok=False); refs={};lines=[]
    def reg(r): refs[r['path']]=r; return r
    def js(name,obj): return reg(write_new(out/name,obj,True))
    def nz(name,values): return reg(save_npz(out/name,values))
    def journal(value):
        raw=(json.dumps(value,sort_keys=True)+'\n').encode();p=out/'original_fixture_calls.jsonl'
        offset=p.stat().st_size if p.exists() else 0
        with p.open('ab') as f:f.write(raw)
        r=dict(path=str(p),offset=offset,length=len(raw),sha256=hashlib.sha256(raw).hexdigest());lines.append(r);return r
    ids=[2,3,4];T=5;nan=np.array([0x7ff8000000000042],np.uint64).view(np.float64)[0]
    old=dict(pose=np.full((T,136),nan),scales=np.zeros(68),shape=np.zeros(45),object_rotation=np.full((T,3,3),nan),object_translation=np.full((T,3),nan),object_scale=np.array(2.5))
    theta=.3;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    for f in ids:
        old['pose'][f]=0.;old['pose'][f,0]=-0.;old['pose'][f,2]=1.5
        old['object_rotation'][f]=R;old['object_translation'][f]=[-0.,0.,10.]
    dense=nz('baseline.npz',old);mask=nz('mask.npz',dict(mask=np.isin(np.arange(T),ids),original_frames=np.array(ids)))
    v=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.],[0.,0.,.1]],np.float32)
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int32)
    obj=reg(write_new(out/'original.obj',(''.join('v '+' '.join(format(float(x),'.17g') for x in row)+'\n' for row in v)+''.join('f '+' '.join(str(int(x)+1) for x in row)+'\n' for row in faces)).encode()))
    mesh=nz('mesh.npz',dict(vertices=v,faces=faces,dimensions=np.array([.1,.1,.1]),unread_container_note=np.array('must not be loaded as a geometry array')))
    K=np.array([[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]])
    raws={};fills={}
    for f in ids:
        raw={k:np.zeros(sh,np.float32) for k,sh in [('mhr_params',(1,204)),('shape_params',(1,45)),('cam_trans',(1,3)),('joints_2d',(1,70,2)),('joints_3d',(1,70,3))]}
        rr=nz(f'raw_{f}.npz',raw);raws[f]=(raw,rr);fills[str(f)]={'frame':f,'raw_sha256':rr['sha256']}
    bp={'producer':'fixture_R0','shape':old['shape'].tolist(),'parents':fills};bpref=js('b_plan.json',bp);parents=[]
    for f in ids:
        raw,rr=raws[f];params=np.zeros(204,np.float32);params[:136]=old['pose'][f]
        b=dict(original_params=params.copy(),adapted_params=params,R=R,t=old['object_translation'][f].copy());br0=nz(f'b_{f}.npz',b)
        inp=dict(pts=f,time_base='1/30',hmr_tensor_hashes={'fixture':f})
        hm=journal(dict(kind='call_end',model_kind='hmr',outcome='returned',episode=0,frame=f,output_identity=legacy_arrays_hash(raw),input_identity=digest(inp['hmr_tensor_hashes'])))
        rigends=[]
        for label,paramkey in [('baseline','original_params'),('adapted','adapted_params')]:
            h=digest(dict(params=legacy_array_hash(b[paramkey]),shape=legacy_array_hash(np.asarray(bp['shape'],np.float32)[None]),plan=digest(bp)))
            rigends.append(journal(dict(kind='call_end',model_kind='rig',outcome='returned',episode=0,frame=f,stage='B:'+label,input_identity=h)))
        fr=js(f'frame_{f}.json',dict(episode=0,source=0,frame=f,raw=rr,hmr_outcome='hmr_valid',producer='fixture_R0',hmr_call_end=hm,input_identity=inp,transform={'K':K.tolist()}))
        br=js(f'b_receipt_{f}.json',dict(frame=f,source=0,output=br0,plan_hash=digest(bp),hmr_fill=fills[str(f)],call_ends=rigends))
        meta=dict(episode=0,source=0,frame=f,raw_frame=f,call_closed=True,dense_raw_sha256=rr['sha256'],raw_sha256=rr['sha256'],numeric_producer='fixture_R0',pts=f,time_base='1/30',input_version='fixture_v1')
        parents.append(dict(raw=rr,frame_receipt=fr,b_params=br0,b_receipt=br,b_plan=bpref,hmr_call_end=hm,rig_call_ends=rigends,meta=meta))
    oldp=dict(q=old['pose'][3].copy(),shape=old['shape'].copy(),scales=old['scales'].copy(),t=old['object_translation'][3].copy(),R=R.copy(),scale=np.array(2.5),K=K)
    newp={k:a.copy() for k,a in oldp.items()};newp['q'][2]-=10*dz;newp['t'][2]+=dz
    op=nz('old_params.npz',oldp);npref=nz('candidate_params.npz',newp)
    source=reg(write_new(out/'synthetic_D_source.py',b'# synthetic fixture; no model runtime\n'))
    model=reg(write_new(out/'synthetic_model_identity.txt',b'not a model; fixture identity only\n'))
    df=js('D_freeze.json',dict(source_tree_sha256=digest({'synthetic_D_source.py':source['sha256']}),files=[source]))
    araw=js('A_original_fixture.json',dict(scope='fixture_only_never_send',statistical_rows=1))
    abinding=dict(run_id='fixture_A',policy_sha256=digest({'fixture_statistics':1}),source_tree_sha256=source['sha256'])
    indexrow=dict(episode=0,source=0,frame=3,proposal_id='synthetic_pulse',target_delta_z_m=dz,pts=3,time_base='1/30',input_version='fixture_v1')
    ai=js('A_index.json',dict(schema='t1_proposal_index_v1',source_ref=araw,binding=abinding,rows=[indexrow]))
    dbinding=dict(run_id='fixture_D',policy_sha256=digest({'fixture_geometry':1}),source_tree_sha256=json.loads(verify(df).read_text())['source_tree_sha256'],proposal_index_sha256=ai['sha256'],model_sha256=model['sha256'])
    native=dict(vertices=np.column_stack([np.linspace(-.1,.1,18439),np.zeros(18439),np.full(18439,10.)])*F*100.,joints=np.column_stack([np.zeros(127),np.zeros(127),np.full(127,10.)])*F*100.)
    newnative={k:a+np.array([0.,0.,dz])*F*100 for k,a in native.items()}
    def rig_record(label,params,values):
        rr=nz(label+'_rig.npz',values)
        forward=dict(params=np.concatenate([params['q'].astype(np.float32),params['scales'].astype(np.float32)])[None],shape=params['shape'].astype(np.float32)[None]);fi=nz(label+'_forward_inputs.npz',forward)
        raw=dict(vertices=values['vertices'][None].copy(),joints=np.concatenate([values['joints'],np.zeros((127,5))],axis=1)[None]);rawref=nz(label+'_raw_forward.npz',raw)
        cc=js(label+'_call.json',dict(schema='t1_persisted_rig_call_v1',coordinate='mhr_native',binding={k:dbinding[k] for k in ('run_id','policy_sha256','source_tree_sha256','model_sha256')},forward_inputs=fi,forward_identity={k:array_identity(v) for k,v in forward.items()},raw_forward=rawref,raw_output_identity=arrays_identity(raw),native_selection='vertices[0],joints[0,:,:3]',start=dict(event='start',kind='rig',call_id=label,key=label,parents={k:array_identity(params[k]) for k in ('q','shape','scales')}),end=dict(event='end',kind='rig',call_id=label,output=rr,output_identity=arrays_identity(values))))
        return rr,cc
    before,bc=rig_record('old',oldp,native);after,ac=rig_record('new',newp,newnative)
    check=geometry_check(oldp,newp,{k:a*F/100 for k,a in native.items()},{k:a*F/100 for k,a in newnative.items()},v,dz)
    row=dict(indexrow,original_dense_ref=dense,old_params=op,candidate_params=npref,old_rig=before,new_rig=after,old_call=bc,new_call=ac,decision=check['decision'],reason=check['reason'],object_mesh=dict(obj=obj,mesh_npz=mesh,vertices_identity=array_identity(v),faces_identity=array_identity(faces),transform='(V*s)@R.T+t'))
    reference_rigs=[dict(episode=0,source=0,frame=3,role=role,binding=dbinding,params_ref=pr,rig_ref=rr,call_ref=cc) for role,pr,rr,cc in [('R0_reference',op,before,bc),('T1_candidate',npref,after,ac)]]
    db=js('D_bundle.json',dict(schema='t1_persisted_geometry_v1',binding=dbinding,model_ref=model,source_freeze=df,reference_rigs=reference_rigs,rows=[row]))
    req=js('required.json',dict(schema='r020_required_frame_plan_v1',episodes=[dict(episode=0,required=ids,sources=[ids],source_map={str(f):0 for f in ids})]))
    adoption=reg(write_new(out/'fixture_adoption.txt',b'fixture approval; no actual production authority'))
    design=js('validation_design.json',dict(schema='t1_validation_design_v1',protocol='prospective_all_A_proposals',proposal_index=ai,source_ref=araw,proposal_count=1,original_sets=ORIGINAL32,adoption=adoption))
    def acceptance(label,artifact,binding,gates):
        evidence={g:js(label+'_'+g+'.json',dict(status='passed',artifact=artifact,binding=binding,scope='fixture_only_never_send')) for g in gates}
        return js(label+'_acceptance.json',dict(schema='t1_stage_acceptance_v1',scope='fixture_only_never_send',status='passed',artifact=artifact,binding=binding,required_plan=req,evidence=evidence,protocol='prospective_all_A_proposals',validation_design=design,original_sets=ORIGINAL32,proposal_count=1))
    aa=acceptance('A',ai,abinding,['statistics','source_closure']); da=acceptance('D',db,dbinding,['parent_binding','rig','projection','visual_all_seven','temporal','resources'])
    keys=[]
    for f in ids:
        for a,n in [(0,46),(3,3),(4,1)]:keys.extend(f't1_000000_{f:06d}_{a}_{i:06d}' for i in range(n))
    for a,n in [(1,23),(2,15),(5,1),(6,32),(7,32)]:keys.extend(f't1_000000_999999_{a}_{i:06d}' for i in range(n))
    rng=np.random.default_rng(93);rng.shuffle(keys);sample=out/'sample.parquet';pd.DataFrame({'row_id':keys}).to_parquet(sample,index=False);sample=reg(reference(sample))
    sf=js('source_freeze.json',source_freeze(ROOT));km=reference(ROOT/'manifests/stock_kit.json')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    c=dict(schema='t1_full_contract_v1',scope='fixture_only_never_send',stage='fixture_only_never_send',policy=POLICY,policy_sha256=digest(POLICY),run_id='fixture_full',source_freeze=sf,source_tree_sha256=json.loads(verify(sf).read_text())['source_tree_sha256'],source_commit=commit,commit_url='https://github.com/HappyGuyLovesTree/nvidia-v2d-track1-cpu/commit/'+commit,model_calls_full=0,kit=str(kit),kit_manifest=km,baseline_rig_model_sha256=model['sha256'],sample=sample,required_plan=req,proposal_index=ai,geometry_bundle=db,a_original=araw,a_acceptance=aa,d_acceptance=da,line_entries=lines,stock_python=str(stock_python),pack_timeout_s=60,resource_policy=dict(seconds=120,output_bytes=67108864,rss_bytes=4294967296,rss_period_s=1,system_period_s=10,rss_gap_s=3,system_unknown_s=20,critical_consecutive=2,hmr=0,rig=0,workers=1),lineage=dict(statistics=abinding,geometry=dbinding,baseline='fixture_R0'),episodes=[dict(episode=0,dense=dense,mask=mask,mesh=obj,numeric_producer='fixture_R0',parents=parents)],artifacts=list(refs.values()))
    c['stock_runtime']=reference(stock_python)
    c['validation_design']=design
    cr=write_new(out/'contract.json',c,True)
    return cr,c,old,newp

class Integration(unittest.TestCase):
    def test_bitwise_negative_zero_nan_payload(self):
        self.assertFalse(bits_equal(np.array([0.]),np.array([-0.])))
        a=np.array([0x7ff8000000000001],np.uint64).view(np.float64);b=np.array([0x7ff8000000000002],np.uint64).view(np.float64)
        self.assertFalse(bits_equal(a,b));self.assertTrue(np.array_equal(a,b,equal_nan=True))
    def test_only_z_slots(self):
        new={k:a.copy() for k,a in OLD.items()};new['pose'][3,2]-=.01;new['object_translation'][3,2]+=.001
        self.assertEqual(full_invariants(OLD,new,[3]),1)
        new['pose'][3,0]=0.
        with self.assertRaises(ContractError):full_invariants(OLD,new,[3])
    def test_nonzero_root_float32_storage_rounding(self):
        c,r,rows,plan,d=load_contract(CONTRACT['path']);x=d['rows'][0]
        a={k:v.astype(np.float32) for k,v in parameters(r,x['old_params']).items()};b={k:v.copy() for k,v in a.items()}
        native=retained_rig(r,x['old_rig'],x['old_call'],parameters(r,x['old_params']),d['binding'])
        v,_=mesh_arrays(c['episodes'][0]['mesh']['path'])
        self.assertEqual(geometry_check(a,b,native,native,v,1e-50)['reason'],'storage_no_change')
        b['q'][2]=np.float32(np.float64(a['q'][2])-.01);b['t'][2]=np.float32(np.float64(a['t'][2])+.001)
        self.assertAlmostEqual(float(b['q'][2]),1.49,places=5)
        self.assertEqual(geometry_check(a,b,native,{k:z+np.array([0.,0.,.001]) for k,z in native.items()},v,.001)['decision'],'accepted')
    def test_rig_bound_output_and_mesh(self):
        c,r,rows,plan,d=load_contract(CONTRACT['path']); row=d['rows'][0];p=parameters(r,row['old_params'])
        self.assertEqual(retained_rig(r,row['old_rig'],row['old_call'],p,d['binding'])['vertices'].shape,(18439,3))
        p['q'][4]=1.
        with self.assertRaises(ContractError):retained_rig(r,row['old_rig'],row['old_call'],p,d['binding'])
        vv,ff=mesh_arrays(C['episodes'][0]['mesh']['path']);bad=copy.deepcopy(row['object_mesh']);bad['transform']='V+t'
        with self.assertRaises(ContractError):mesh_binding(r,bad,C['episodes'][0]['mesh'],vv,ff)
    def test_f4_wrong_parent_closed_flag(self):
        c,r,rows,plan,d=load_contract(CONTRACT['path']);p=copy.deepcopy(c['episodes'][0]['parents'][0]);p['meta']['call_closed']=False
        with self.assertRaises(ContractError):verify_parent(r,p,0,0,2,OLD,c['line_entries'])
        p=copy.deepcopy(c['episodes'][0]['parents'][0]);p['hmr_call_end']=c['episodes'][0]['parents'][1]['hmr_call_end']
        with self.assertRaises(ContractError):verify_parent(r,p,0,0,2,OLD,c['line_entries'])
    def test_raw_forward_native_selection_mismatch(self):
        c,r,rows,plan,d=load_contract(CONTRACT['path']);row=d['rows'][0];params=parameters(r,row['old_params']);call=r.json(row['old_call']);raw=r.npz(call['raw_forward']);raw['vertices'][0,0,0]+=1.
        nr=save_npz(OUT/'bad_native_raw.npz',raw);call['raw_forward']=nr;call['raw_output_identity']=arrays_identity(raw);nc=write_new(OUT/'bad_native_call.json',call)
        rr=Reader([*c['artifacts'],nr,nc])
        with self.assertRaisesRegex(ContractError,'native selection'):retained_rig(rr,row['old_rig'],nc,params,d['binding'])
    def test_actual_D_call_aliases_and_protocol_scope(self):
        c,r,rows,plan,d=load_contract(CONTRACT['path']);row=d['rows'][0];params=parameters(r,row['old_params']);call=r.json(row['old_call']);rawref=call['raw_forward']
        for k in ('forward_inputs','forward_identity','raw_forward','raw_output_identity','native_selection'):call.pop(k)
        p=np.concatenate([params['q'].astype(np.float32),params['scales'].astype(np.float32)]);sh=params['shape'].astype(np.float32)
        call.update(raw_forward_ref=rawref,native_joint_selection='all127 first3 coordinates from raw[0,:,:3]',camera_transform='F=diag(1,-1,-1); camera_m=native*F/100')
        call['start']['parents'].update(actual_params204=array_identity(p),actual_shape45=array_identity(sh),params_sha256=legacy_array_hash(p),shape_sha256=legacy_array_hash(sh),input_contract_sha256='fixture_input',**{k:d['binding'][k] for k in ('source_tree_sha256','policy_sha256','model_sha256')})
        original_end=call['end'].copy();start_raw=(json.dumps(call['start'])+'\n').encode();end_raw=(json.dumps(original_end)+'\n').encode();journal=OUT/'actual_D_style_journal.jsonl';write_new(journal,start_raw+end_raw)
        slices=[dict(call_id=call['start']['call_id'],start=dict(path=str(journal),offset=0,length=len(start_raw),sha256=hashlib.sha256(start_raw).hexdigest()),end=dict(path=str(journal),offset=len(start_raw),length=len(end_raw),sha256=hashlib.sha256(end_raw).hexdigest()))]
        cr=write_new(OUT/'actual_D_style_call.json',call);rr=Reader([*c['artifacts'],cr]);retained_rig(rr,row['old_rig'],cr,params,d['binding'],'fixture_input',slices)
        with self.assertRaises(ContractError):retained_rig(rr,row['old_rig'],cr,params,d['binding'],'wrong_contract',slices)
        bad=copy.deepcopy(c);bad['scope']='official'
        with self.assertRaisesRegex(ContractError,'supervised worker'):assemble(bad,r,plan,d,OUT/'must_not_create')
        design=r.json(c['validation_design']);design['original_sets'][0]['status']='passed';dr=write_new(OUT/'bad_validation_design.json',design)
        bad=copy.deepcopy(c);bad['validation_design']=dr;bad['artifacts'].append(dr);cr=write_new(OUT/'bad_validation_contract.json',bad)
        with self.assertRaisesRegex(ContractError,'old32'):load_contract(cr['path'])
    def test_native_bundle_aliases_without_following_refs(self):
        from v2d_t1.native_adapter import normalize_geometry
        d=json.loads(verify(C['geometry_bundle']).read_text());x=copy.deepcopy(d['rows'][0]);mb=x.pop('object_mesh');x.update(key_id='fixture_key',decision='accepted',reason='geometric_acceptance_only',gate={'accepted':True},mesh_binding=dict(obj_ref=mb['obj'],mesh_npz_ref=mb['mesh_npz'],vertices_identity=mb['vertices_identity'],faces_identity=mb['faces_identity'],coordinate='local_metres',transform='(vertices*scale)@R.T+t; original scale/R and actual stored old/candidate t',old_transform_params=x['old_params'],candidate_transform_params=x['candidate_params']))
        n=dict(schema=d['schema'],coordinate='mhr_native',binding=d['binding'],rows=[x],all_old_reference_rig_outputs=[dict(parents={'key_id':'fixture_key','role':'R0_reference'},output=x['old_rig'],call_receipt=x['old_call'])])
        ctx=dict(schema='t1_D_adapter_context_v1',binding=d['binding'],model_ref=d['model_ref'],source_freeze=d['source_freeze'],input_contract={'sha256':'fixture_input'},reference_catalog=[dict(key_id='fixture_key',episode=x['episode'],source=x['source'],frame=x['frame'],params_ref=x['old_params'],pts=x['pts'],time_base=x['time_base'],input_version=x['input_version'])])
        normalized=normalize_geometry(n,ctx);self.assertEqual(normalized['binding'],n['binding']);self.assertEqual(normalized['rows'][0]['reason'],'accepted');self.assertEqual(normalized['rows'][0]['object_mesh']['obj'],mb['obj'])
    def test_single_point_projection_and_nonunit_scale(self):
        c,r,rows,plan,d=load_contract(CONTRACT['path']);x=d['rows'][0];a=parameters(r,x['old_params']);b=parameters(r,x['candidate_params']);before=retained_rig(r,x['old_rig'],x['old_call'],a,d['binding']);after=retained_rig(r,x['new_rig'],x['new_call'],b,d['binding']);v,_=mesh_arrays(c['episodes'][0]['mesh']['path'])
        self.assertEqual(geometry_check(a,b,before,after,v,x['target_delta_z_m'])['decision'],'accepted')
        after['vertices'][123,0]+=1.
        self.assertEqual(geometry_check(a,b,before,after,v,x['target_delta_z_m'])['reason'],'rig')
        a['t'][2]=.051;b['t'][2]=.052
        for p in (a,b):p['scale']=np.array(100.)
        self.assertNotEqual(geometry_check(a,b,before,{k:z+np.array([0,0,.001]) for k,z in before.items()},v,.001)['decision'],'accepted')
    def test_fixture_never_send_and_argv(self):
        with self.assertRaises(ContractError):gate.evidence(dict(schema='t1_cdh_contract_v1',scope='fixture_only_never_send',stage='cdh_send_authorized'))
        c=dict(uploader={'path':'/fixed/uploader'},run_id='fixture',candidate={'sha256':'a'*64});argv=gate.final_argv(c,Path('/fixture/output'))
        self.assertEqual(argv[argv.index('--only')+1],'cd_h_cm');self.assertEqual(argv.count('--only'),1)
    def test_final_send_argv_and_exclusive_intent_mock_only(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        out=OUT/'mock_send';out.mkdir();src=write_new(out/'source.bin',b'fixture is not parquet')
        snap=write_new(out/'candidate.parquet',b'fixture is not parquet',True)
        c=dict(uploader={'path':'/fixture/not-executed'},candidate=src,sample=src,commit_url='fixture',run_id='fixture_send',policy_sha256='fixture',source_tree_sha256='fixture',source_commit='fixture')
        bound=dict(scope='official',validation_mode='complete',candidate=src,snapshot=snap,contract_sha256=digest(c),argv=gate.final_argv(c,out),**{k:c[k] for k in gate.BOUND})
        # Isolate the final F1 boundary; evidence/parquet/network are explicitly mocked.
        with patch.object(gate,'evidence',return_value={}),patch.object(gate,'validate_candidate',return_value={}),patch.object(gate,'sample_rows',return_value=[]),patch.object(gate.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='mock',stderr='')) as call:
            bad=copy.deepcopy(bound);bad['argv']+=['--only','cd_o_cm'];write_new(out/'binding.json',bad,True)
            with self.assertRaises(ContractError):gate.send(c,bad,out)
            self.assertEqual(call.call_count,0)
        out2=OUT/'mock_send_legal';out2.mkdir();snap=write_new(out2/'candidate.parquet',b'fixture is not parquet',True)
        bound.update(snapshot=snap,argv=gate.final_argv(c,out2));write_new(out2/'binding.json',bound,True)
        with patch.object(gate,'evidence',return_value={}),patch.object(gate,'validate_candidate',return_value={}),patch.object(gate,'sample_rows',return_value=[]),patch.object(gate.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='mock',stderr='')) as call:
            gate.send(c,bound,out2);self.assertEqual(call.call_count,1)
            with self.assertRaises(FileExistsError):gate.send(c,bound,out2)
            self.assertEqual(call.call_count,1)
    def test_zero_actual_change_denied(self):
        cr,c,old,new=fixture(OUT/'zero_change',KIT,STOCK,dz=0.);c,r,rows,plan,d=load_contract(cr['path'])
        with self.assertRaisesRegex(ContractError,'zero actual change'):assemble(c,r,plan,d,OUT/'zero_view')
    def test_real_supervised_stock_fixture_roundtrip(self):
        argv=[str(SUPERVISOR),*(['-O']*sys.flags.optimize),'-B',str(ROOT/'tools/t1.py'),'run','--contract',CONTRACT['path'],'--output',str(OUT/'supervised_fixture'),'--worker-python',str(STOCK)]
        cp=subprocess.run(argv,text=True,capture_output=True,timeout=100)
        write_new(OUT/'launcher_process.json',dict(argv=argv,exit=cp.returncode,stdout=cp.stdout,stderr=cp.stderr))
        self.assertEqual(cp.returncode,0,cp.stderr[-3000:])
        self.assertTrue((OUT/'supervised_fixture/supervisor_receipt.json').is_file())
        receipt=json.loads((OUT/'supervised_fixture/supervisor_receipt.json').read_text());self.assertEqual(receipt['status'],'completed');self.assertIsNotNone(receipt['reap'])
        result=json.loads((OUT/'supervised_fixture/work/full_result.json').read_text());self.assertEqual(result['status'],'fixture_only_never_send');self.assertEqual(result['model_calls'],0);self.assertGreater(result['decoded_changed_values'],0)
        candidate=Path(result['candidate']['path']);df=pd.read_parquet(candidate);bad=OUT/'bad_order.parquet';df.iloc[::-1].to_parquet(bad,index=False)
        with self.assertRaises(ValueError):roundtrip(C,OUT/'supervised_fixture/work/stock_view',bad)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--kit',required=True);ap.add_argument('--stock-python',required=True);ap.add_argument('--supervisor-python',required=True);a=ap.parse_args()
    OUT=Path(a.output).resolve();OUT.mkdir(parents=True,exist_ok=False);KIT=Path(a.kit).resolve();STOCK=Path(a.stock_python).absolute();SUPERVISOR=Path(a.supervisor_python).absolute()
    CONTRACT,C,OLD,NEW=fixture(OUT/'inputs',KIT,STOCK)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Integration);res=unittest.TextTestRunner(verbosity=2).run(suite)
    write_new(OUT/'result.json',dict(schema='t1_synthetic_integration_tests_v1',scope='fixture_only_never_send',tests=res.testsRun,failures=len(res.failures),errors=len(res.errors),status='passed' if res.wasSuccessful() else 'failed',real_model_calls=0,real_prediction_files_read=0,real_uploads=0,synthetic_stock_fixture=True))
    sys.exit(0 if res.wasSuccessful() else 1)
