"""Model-free full conversion, immutable stock view and official eight-array roundtrip."""
from pathlib import Path
import argparse, json, os, sys
import numpy as np
import pandas as pd
from v2d_public.inputs import SAMPLE_SHA, required_from_keys, sample_rows
from v2d_public.packing import verify_kit, validate_candidate
from v2d_public.common import run_logged
from .core import *
from .lineage import DENSE, PARAMS, POLICY, parameters, verify_parent, retained_rig, geometry_check

def key(row): return (row['episode'],row['source'],row['frame'])

def contract_header(path):
    c=json.loads(Path(path).read_text())
    need(c['schema']=='t1_full_contract_v1' and c['scope'] in ('official','fixture_only_never_send'), 'full contract schema/scope')
    need(c['policy']==POLICY and c['policy_sha256']==digest(POLICY), 'fixed integration policy')
    need(c['stage']==('full_generation_authorized' if c['scope']=='official' else 'fixture_only_never_send'), 'full stage')
    need(c['model_calls_full']==0, 'full converter must not run models')
    rp=c['resource_policy']
    need(all(np.isfinite(rp[k]) and rp[k]>0 for k in ('seconds','output_bytes','rss_bytes')), 'finite positive resource bounds')
    need({k:rp[k] for k in ('rss_period_s','system_period_s','rss_gap_s','system_unknown_s','critical_consecutive')}==dict(rss_period_s=1,system_period_s=10,rss_gap_s=3,system_unknown_s=20,critical_consecutive=2),'fixed observation/stop policy')
    need(c['stock_python']==c['stock_runtime']['path'] and verify(c['stock_runtime']), 'pinned stock/worker runtime')
    verify_freeze(c['source_freeze'],c['source_tree_sha256'])
    need(c['commit_url']=='https://github.com/HappyGuyLovesTree/nvidia-v2d-track1-cpu/commit/'+c['source_commit'], 'source URL')
    if c['scope']=='official':
        git_identity(Path(__file__).resolve().parents[1], c['source_commit'])
        need(c['resource_policy']['seconds']<=3600 and c['resource_policy']['output_bytes']<=1073741824 and c['resource_policy']['rss_bytes']<=4294967296, 'full resource bounds')
    need(c['resource_policy']['workers']==1 and c['resource_policy']['hmr']==c['resource_policy']['rig']==0, 'full worker/model budget')
    return c

ORIGINAL32=[dict(name='development20',frames=20,proposals=0,status='no_trigger_not_applicable'),dict(name='independent12',frames=12,proposals=0,status='no_trigger_not_applicable')]
ADOPTION_SHA='16433f53caecd146dfae4d967e70af21a9669d13011fa73addd71abefddd21b3'

def load_contract(path):
    c=contract_header(path)
    r=Reader(c['artifacts']); r.path(c['sample']); rows=sample_rows(c['sample']['path']); required=required_from_keys(rows)
    p=r.json(c['required_plan']); need(p['schema']=='r020_required_frame_plan_v1', 'required plan schema')
    need([e['episode'] for e in p['episodes']]==sorted(required), 'required episode order')
    count=0
    for e in p['episodes']:
        ids=e['required']; groups=e['sources']
        need(ids==required[e['episode']] and ids==sorted(set(ids)), 'sample required-frame mapping')
        need(groups and all(g and g==list(range(g[0],g[-1]+1)) for g in groups), 'source continuity')
        need([f for g in groups for f in g]==ids and e['source_map']=={str(f):i for i,g in enumerate(groups) for f in g}, 'exact source partition')
        count+=len(groups)
    if c['scope']=='official':
        need(sha(c['sample']['path'])==SAMPLE_SHA and len(rows)==740780 and len(required)==30 and sum(map(len,required.values()))==9877 and count==65, 'official complete coverage')
    need([e['episode'] for e in c['episodes']]==sorted(required), 'baseline episode order')
    if c['scope']=='official':
        for ep,req in zip(c['episodes'],p['episodes']):
            original=r.json(ep['episode_input'],['/episode','/required','/sources','/source_map'])
            need({k:original[k] for k in ('episode','required','sources','source_map')}==req,'actual original required/source metadata')
            need(r.legacy_digest(ep['episode_input'])==ep['input_legacy_digest'],'original input lexical digest')
    a=r.json(c['proposal_index']); d=r.json(c['geometry_bundle'])
    if 'all_old_reference_rig_outputs' in d:
        from .native_adapter import normalize_geometry
        context=r.json(c['d_adapter_context']); ip=r.json(context['input_plan']); ic=r.json(context['input_contract'])
        need(ic['input_plan']==context['input_plan'] and ip['source_freeze']==context['source_freeze']==ic['source_freeze'] and ip['rig_asset']==context['model_ref'],'actual D plan/model/source context')
        need(ip['proposal_index']==d['proposal_index']==c['proposal_index'] and ip['proposal_manifest']==d['proposal_source']==c['a_original'],'actual D proposal context')
        pm={x['key_id']:x for x in ip['rows']}
        for entry in context['reference_catalog']:
            row=pm[entry['key_id']]
            need((row['episode'],row['source'],row['frame'])==key(entry),'actual D reference plan mapping')
        d=normalize_geometry(d,context)
    need(a['schema']=='t1_proposal_index_v1' and d['schema']=='t1_persisted_geometry_v1', 'A/D schema')
    need(d['binding']['proposal_index_sha256']==c['proposal_index']['sha256'], 'D/A binding')
    need(d['binding']['model_sha256']==c['baseline_rig_model_sha256'], 'D model differs from approved R0 rig')
    need(r.path(d['model_ref']) and d['model_ref']['sha256']==d['binding']['model_sha256'], 'D actual model asset binding')
    df=r.json(d['source_freeze'])
    need(df['source_tree_sha256']==d['binding']['source_tree_sha256'] and df['files'], 'D source freeze binding')
    for source_ref in df['files']: r.path(source_ref)
    for config_ref in df.get('configuration_refs',[]):r.path(config_ref)
    r.path(a['source_ref']); need(a['source_ref']==c['a_original'], 'actual A source ref')
    need(len({key(x) for x in a['rows']})==len(a['rows']) and len({key(x) for x in d['rows']})==len(d['rows']), 'duplicate proposal')
    ad={key(x):x for x in a['rows']}; dd={key(x):x for x in d['rows']}
    need(set(ad)==set(dd), 'all A proposals including rejects required')
    need(all(ad[k]['proposal_id']==dd[k]['proposal_id'] and ad[k]['target_delta_z_m']==dd[k]['target_delta_z_m'] for k in ad), 'proposal changed')
    if c['scope']=='official':
        need(len(a['rows'])==7 and c['a_original']['sha256']==c['approved_a_sha256']=='26a932c6e7737d80433c9003525826281fe2fb129c74b14247ddf03db468d64d', 'fixed seven actual A proposals')
    design=r.json(c['validation_design'])
    need(design['schema']=='t1_validation_design_v1' and design['protocol']=='prospective_all_A_proposals' and design['proposal_index']==c['proposal_index'] and design['source_ref']==c['a_original'] and design['proposal_count']==len(a['rows']) and design['original_sets']==ORIGINAL32, 'exact a3 prospective protocol; old32 remain zero/NA')
    r.path(design['adoption'])
    if c['scope']=='official':need(design['adoption']['sha256']==ADOPTION_SHA,'actual a3 adoption pin')
    for name,target in [('a_acceptance',c['proposal_index']),('d_acceptance',c['geometry_bundle'])]:
        v=r.json(c[name])
        if name=='a_acceptance' and v.get('schema')!='t1_stage_acceptance_v1':
            need(v['status']=='passed' and v['proposal_index']==c['proposal_index'] and v['source_ref']==c['a_original'],'original independent A index validation')
            continue
        if name=='d_acceptance' and v.get('schema')=='t1_independent_D_acceptance_v1':
            need(v['status']=='passed' and v['scope']=='T1_D_all7_statistically_selected_prospective_small_set_only' and v['quality_pass'] is True,'actual B8 prospective small-set acceptance')
            need(v['geometry_bundle']==c['geometry_bundle'] and v['proposal_index']==c['proposal_index'] and v['proposal_source']==c['a_original'] and v['binding']==d['binding'],'actual B8 source binding')
            need(v['proposals']==7 and v['accepted_updates']==c['expected_accepts'] and v['rejected_whole_frame_R0']==7-c['expected_accepts'] and v['original_labels_changed'] is False,'actual B8 complete denominator')
            need(c['B8_manifest']['sha256']=='51a374b86270ec88e2b76a64f73949be2a6687e9663b21c34a2a6502117f0582','approved B8 manifest')
            safe=r.json(c['B8_manifest'])['safe_refs'];need(c[name] in safe and len(safe)==9,'B8 safe evidence scope')
            for rr in safe:r.path(rr)
            expected={'provenance':'passed','rig':'passed','projection':'passed_for_final_adopted_subset_rejected4_retained','all7_visible':'passed_no_clear_new_degradation_of3_actual_changes','temporal':'passed_on2_edges_in1_segment_other12_NA','resources':'passed_with_initial_adapter_stop_preserved'}
            for gate,status in expected.items():
                need(v['gates'][gate]['status']==status and v['gates'][gate]['evidence'] in safe,'B8 exact gate/proof scope '+gate)
            continue
        need(v['schema']=='t1_stage_acceptance_v1' and v['artifact']==target, 'independent stage artifact binding')
        need(v['status']=='passed' and v['scope']==c['scope'] and v['required_plan']==c['required_plan'], 'independent stage scope/plan')
        need(v['binding']==(a['binding'] if name=='a_acceptance' else d['binding']), 'independent producer binding')
        if name=='d_acceptance':need(v['protocol']==design['protocol'] and v['validation_design']==c['validation_design'] and v['original_sets']==ORIGINAL32 and v['proposal_count']==len(a['rows']), 'B8 prospective all-A protocol and original32 state')
        for gate in (['statistics','source_closure'] if name=='a_acceptance' else ['parent_binding','rig','projection','visual_all_seven','temporal','resources']):
            ev=r.json(v['evidence'][gate]); applicable=ev['status']=='passed'
            if gate=='temporal' and ev['status']=='not_applicable':
                applicable=ev['eligible_edges']==0 and ev['all_edges_recorded'] is True and ev['protocol']==design['protocol'] and ev['validation_design']==c['validation_design']
            need(applicable and ev['artifact']==target and ev['binding']==v['binding'], 'independent evidence binding '+gate)
    need(c['lineage']['statistics']==a['binding'] and c['lineage']['geometry']==d['binding'], 'producer layer collapse')
    used_slices={digest(x) for ep in c['episodes'] for parent in ep['parents'] for x in [parent['hmr_call_end'],*parent['rig_call_ends']]}
    need(used_slices and used_slices=={digest(x) for x in c['line_entries']} and len(used_slices)==len(c['line_entries']),'exact nonempty original call slice whitelist')
    if c['scope']=='official':
        pairs=c['source_call_pairs'];need({digest(x['end']) for x in pairs}==used_slices and len(pairs)==len(used_slices),'exact original start/end set')
        for pair in pairs:
            start=read_slice(pair['start']);end=read_slice(pair['end'])
            need(start['kind']=='call_start' and end['kind']=='call_end' and start['call_id']==end['call_id']==pair['call_id'],'original selected call closure')
            for field in ('model_kind','episode','frame','attempt','stage','ordinal','input_identity'):need(start[field]==end[field],'original call start/end parent mismatch')
    references=d['reference_rigs']; need(len(references)<=28, 'persistent rig inventory above planned bound')
    inventory={}
    for entry in references:
        need(entry['role'] in ('R0_reference','T1_candidate'), 'rig inventory role')
        k=(entry['role'],*key(entry)); need(k not in inventory, 'duplicate rig inventory key')
        binding=entry['binding']; need(binding['model_sha256']==d['binding']['model_sha256'], 'reused rig model differs')
        if entry['role']=='T1_candidate': need(binding==d['binding'],'new rig must retain actual D producer')
        elif binding!=d['binding']:
            reuse=r.json(entry['reuse_acceptance'])
            need(reuse['status']=='passed' and reuse['binding']==binding and reuse['call']==entry['call_ref'] and reuse['output']==entry['rig_ref'] and reuse['params']==entry['params_ref'],'old cache independent reuse binding')
            oldfreeze=r.json(entry['source_freeze']); need(oldfreeze['source_tree_sha256']==binding['source_tree_sha256'] and oldfreeze['files'],'old cache source freeze')
            for oldsource in oldfreeze['files']:r.path(oldsource)
        pp=parameters(r,entry['params_ref']); retained_rig(r,entry['rig_ref'],entry['call_ref'],pp,binding,d.get('input_contract_sha256'),d.get('call_slices')); inventory[k]=entry
    for row in d['rows']:
        for role,prefix in [('R0_reference','old'),('T1_candidate','new')]:
            entry=inventory[(role,*key(row))]
            need((entry['params_ref'],entry['rig_ref'],entry['call_ref'])==(row['old_params' if prefix=='old' else 'candidate_params'],row[prefix+'_rig'],row[prefix+'_call']), 'proposal/inventory rig mismatch')
    need(sum(e['role']=='R0_reference' for e in references)<=21 and sum(e['role']=='T1_candidate' for e in references)==len(d['rows']), 'rig inventory count')
    need(c['kit_manifest']==reference(Path(__file__).resolve().parents[1]/'manifests/stock_kit.json'),'fixed source kit manifest')
    verify_kit(c['kit'],json.loads(verify(c['kit_manifest']).read_text()))
    return c,r,rows,p,d

def full_invariants(old,new,allowed):
    need(set(old)==set(new)==set(DENSE), 'six arrays')
    for k in DENSE:
        if k not in ('pose','object_translation'): need(bits_equal(old[k],new[k]), 'immutable field '+k)
    for k in ('pose','object_translation'):
        a=old[k]; b=new[k]; need(a.dtype==b.dtype and a.shape==b.shape, 'time array schema')
        x=b.copy()
        for f in allowed: x[f,2]=a[f,2]
        need(bits_equal(a,x), 'changed non-permitted slot or inherited row')
    actual={f for f in allowed if not bits_equal(old['pose'][f],new['pose'][f]) or not bits_equal(old['object_translation'][f],new['object_translation'][f])}
    need(actual==set(allowed), 'accepted update did not change stored numeric fields')
    return len(actual)

def mesh_arrays(path):
    import trimesh
    need(not any(line.lstrip().startswith('mtllib ') for line in Path(path).read_text().splitlines()),'external OBJ materials are outside the mesh contract')
    m=trimesh.load(path,force='mesh',process=False,maintain_order=True,skip_materials=True)
    v=np.asarray(m.vertices,np.float64); f=np.asarray(m.faces,np.int64)
    need(v.ndim==2 and v.shape[1]==3 and len(v)>0 and np.isfinite(v).all(), 'actual mesh vertices')
    need(f.ndim==2 and f.shape[1]==3 and f.dtype.kind in 'iu' and (f>=0).all() and (f<len(v)).all(), 'actual mesh faces')
    good=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)>0
    need(good.any(), 'empty actual mesh surface')
    # Original source OBJ vertices, including unused genuine vertices; no budget padding.
    return v,f

def mesh_binding(reader, declared, original_ref, vertices, faces):
    need(declared['obj']==original_ref and declared['transform']=='(V*s)@R.T+t', 'D actual object mesh/transform binding')
    reader.path(declared['obj']); src=reader.npz(declared['mesh_npz'],['vertices','faces'],subset=True)
    need('vertices' in src and 'faces' in src, 'original object mesh NPZ fields')
    need(np.array_equal(src['vertices'],vertices) and np.array_equal(src['faces'],faces), 'OBJ versus D source mesh')
    need(declared['vertices_identity']==array_identity(src['vertices']) and declared['faces_identity']==array_identity(src['faces']), 'D original consumed mesh dtype identity')
    return src['vertices']

def assemble(c,r,plan,d,out,supervision_root=None):
    from v2d_public.packing import check_view
    out=Path(out)
    if c['scope']=='official':
        need(supervision_root is not None,'official assembly requires supervised worker')
        worker_authority(c,Path(supervision_root)/'work',Path(supervision_root))
        need(out==Path(supervision_root)/'work/stock_view','official stock view output')
    out.mkdir(parents=True,exist_ok=False); accepted=[]; diagnostics=[]; lineage=[]
    proposals={key(x):x for x in d['rows']}; consumed=set()
    indexes={key(x):x for x in r.json(c['proposal_index'])['rows']}
    old_refs={key(x):x for x in d['reference_rigs'] if x['role']=='R0_reference'}; seen_refs=set()
    for ep,req in zip(c['episodes'],plan['episodes']):
        e=ep['episode']; ids=req['required']; old=r.npz(ep['dense'],DENSE); mask=r.npz(ep['mask'],['mask','original_frames'])
        r.path(ep['mesh']); new={k:v.copy() for k,v in old.items()}; changes=[]
        need(mask['mask'].dtype==np.bool_ and mask['original_frames'].tolist()==ids and np.flatnonzero(mask['mask']).tolist()==ids, 'mask frame coverage')
        need([x['meta']['frame'] for x in ep['parents']]==ids, 'all baseline parents required')
        vertices,faces=mesh_arrays(ep['mesh']['path'])
        for parent in ep['parents']:
            f=parent['meta']['frame']; source=req['source_map'][str(f)]; k=(e,source,f)
            need(parent['meta']['numeric_producer']==ep['numeric_producer'], 'baseline episode producer')
            current=verify_parent(r,parent,e,source,f,old,c['line_entries'])
            raw_frame=parent['meta']['raw_frame']
            need(req['source_map'].get(str(raw_frame))==source,'raw parent outside original source')
            if k in old_refs:
                rp=parameters(r,old_refs[k]['params_ref']); seen_refs.add(k)
                for field,target in [('q',old['pose'][f]),('t',old['object_translation'][f]),('R',old['object_rotation'][f]),('scale',old['object_scale']),('shape',old['shape']),('scales',old['scales'])]:
                    need(bits_equal(rp[field],target),'R0 trajectory reference differs from baseline '+field)
            if k not in proposals: continue
            need(current, 'borrowed parent cannot be T1 proposal')
            row=proposals[k]; consumed.add(k); a=parameters(r,row['old_params']); b=parameters(r,row['candidate_params'])
            consumed_vertices=mesh_binding(r,row['object_mesh'],ep['mesh'],vertices,faces)
            for field in ('pts','time_base','input_version'):
                need(row[field]==parent['meta'][field], 'D input plan/baseline temporal identity '+field)
            original_K=np.asarray(r.json(parent['frame_receipt'],['/transform/K'])['transform']['K'],np.float64)
            need(np.array_equal(original_K,a['K']), 'original K binding')
            for field,target in [('q',old['pose'][f]),('t',old['object_translation'][f]),('R',old['object_rotation'][f]),('scale',old['object_scale']),('shape',old['shape']),('scales',old['scales'])]:
                need(bits_equal(a[field],target), 'D/baseline parameter identity '+field)
            before=retained_rig(r,row['old_rig'],row['old_call'],a,old_refs[k]['binding'],d.get('input_contract_sha256'),d.get('call_slices'))
            after=retained_rig(r,row['new_rig'],row['new_call'],b,d['binding'],d.get('input_contract_sha256'),d.get('call_slices'))
            check=geometry_check(a,b,before,after,consumed_vertices,row['target_delta_z_m'])
            need((check['decision'],check['reason'])==(row['decision'],row['reason']), 'D geometric decision disagrees')
            need(row['original_dense_ref']==ep['dense'],'D exact original dense ref')
            if check['decision']=='accepted':
                new['pose'][f]=b['q']; new['object_translation'][f]=b['t']; changes.append(f); accepted.append(k)
            diagnostics.append(dict(episode=e,source=source,frame=f,decision=check,old_call=row['old_call'],new_call=row['new_call']))
        full_invariants(old,new,changes)
        nr=save_npz(out/'episodes'/f'episode_{e:06d}.npz',new)
        with np.load(nr['path'],allow_pickle=False) as z:persisted={k:z[k] for k in z.files}
        full_invariants(old,persisted,changes)
        mr=write_new(out/'masks'/f'episode_{e:06d}.npz',r.path(ep['mask']).read_bytes(),True)
        mesh=write_new(out/'episodes'/f'episode_{e:06d}_object.obj',r.path(ep['mesh']).read_bytes(),True)
        lineage.append(dict(episode=e,baseline_numeric_producer=ep['numeric_producer'],baseline_dense=ep['dense'],dense=nr,mask=mr,mesh=mesh,accepted_frames=changes))
    need(consumed==set(proposals), 'proposal outside full required/source map')
    need(seen_refs==set(old_refs), 'R0 trajectory reference outside full source/required map')
    need(accepted, 'zero actual change: do not build candidate')
    if c['scope']=='official':need(len(accepted)==c['expected_accepts'],'actual accepted subset differs from B8')
    # Mature stock-view checks are reused with fixture scope; full scope coverage was enforced above.
    check_view(out/'episodes',sample_rows(c['sample']['path']),masks=out/'masks',scope='fixture')
    result=dict(schema='t1_stock_view_v1',scope=c['scope'],accepted=[list(k) for k in accepted],diagnostics=diagnostics,
                lineage=dict(c['lineage'],converter=dict(run_id=c['run_id'],source_commit=c['source_commit'],source_tree_sha256=c['source_tree_sha256'],policy_sha256=c['policy_sha256'])),episodes=lineage)
    write_new(out/'provenance.json',result,True); r.finish()
    for sl in c['line_entries']:read_slice(sl)
    return result

def roundtrip(c,view,candidate):
    sys.path.insert(0,str(Path(c['kit']).resolve()))
    from v2dlb import mhr_submission as ms
    from v2dlb.mesh_budget import budget_mesh
    from v2dlb.mesh_common import _mesh_faces
    rows=sample_rows(c['sample']['path']); validate_candidate(candidate,rows,c['commit_url'])
    result=pd.read_parquet(candidate); keys=result.row_id.str.split('_',expand=True); reports=[]
    for e,ids in required_from_keys(rows).items():
        take=keys[1].astype(int)==e; block=result.loc[take,['x','y','z']].copy(); p=keys.loc[take]
        block['episode_index']=p[1].astype(int); block['frame_index']=p[2].astype(int)
        block['array_name']=p[3].astype(int).map(dict(enumerate(ms.array_order()))); block['point_index']=p[4].astype(int)
        arrays=ms._to_arrays(block,['x','y','z'])
        with np.load(Path(view)/'episodes'/f'episode_{e:06d}.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
        for name,count,key0 in [('mhr_pose',136,'pose'),('mhr_scales',68,'scales'),('mhr_shape',45,'shape')]:
            expected=d[key0][ids] if key0=='pose' else d[key0]
            need(bits_equal(ms._unpack(arrays[name],count,name),expected.astype(np.float64)), 'official parameter/tail roundtrip')
        need(bits_equal(arrays['object_rotation'],d['object_rotation'][ids].astype(np.float64)) and bits_equal(arrays['object_translation'][:,0],d['object_translation'][ids].astype(np.float64)), 'R/t roundtrip')
        scale=ms._object_scale(arrays['object_scale']); need(scale==float(d['object_scale']), 'scale roundtrip')
        vv=arrays['object_mesh_vertices']; ff=_mesh_faces(arrays['object_mesh_faces'],len(vv))
        bv,bf=budget_mesh(Path(view)/'episodes'/f'episode_{e:06d}_object.obj',len(ff),len(vv))
        need(bits_equal(vv,bv.astype(np.float64)) and np.array_equal(ff,bf), 'real mesh/budget/padding roundtrip')
        projected=ms._rotations(arrays['object_rotation']); need(float(np.abs(projected-d['object_rotation'][ids]).max())<=1e-6, 'decoded rotation precision')
        # Include the actual scale on both sides, for every required frame and all budget vertices.
        maxerr=0.
        for j,f in enumerate(ids):
            direct=float(d['object_scale'])*(vv@d['object_rotation'][f].T)+d['object_translation'][f]
            decoded=scale*(vv@projected[j].T)+arrays['object_translation'][j,0]
            maxerr=max(maxerr,float(np.abs(direct-decoded).max()))
        need(maxerr<=1e-4,'decoded object transform precision')
        reports.append(dict(episode=e,frames=len(ids),all_eight_arrays=True,object_transform_max_m=maxerr))
    return reports

def worker_authority(c,out,root):
    need(root is not None, 'live supervisor required before array access')
    root=Path(root);out=Path(out)
    origin=json.loads((root/'origin.json').read_text())
    need(origin['authority']==dict(contract_sha256=digest(c),stage=c['stage']) and origin['fixture']==(c['scope']!='official'), 'supervisor authority')
    child=json.loads((root/'child_identity.json').read_text())
    need(child['pid']==os.getpid() and os.getppid()==origin['supervisor']['pid'] and out==root/'work', 'live supervisor parent/output')
    expected=[c['stock_runtime']['path'],*(['-O']*sys.flags.optimize),'-B',str(Path(__file__).resolve().parents[1]/'tools/t1.py'),'worker','--contract',str(root/'contract.json'),'--output',str(out),'--supervision-root',str(root)]
    need(origin['argv']==expected,'fixed supervised worker argv')
    return origin

def worker(contract_path,out,supervision_root):
    c=contract_header(contract_path);out=Path(out);root=Path(supervision_root);origin=worker_authority(c,out,root)
    def deny_models(event,args):
        if event=='import' and args[0].split('.')[0] in ('torch','onnxruntime','tensorflow','av','cv2'): need(False,'model/video import denied')
    sys.addaudithook(deny_models)
    c,r,rows,plan,d=load_contract(contract_path)
    view=out/'stock_view'; lineage=assemble(c,r,plan,d,view,root)
    candidate=out/'candidate.parquet'
    argv=[c['stock_python'],'-B',str(Path(c['kit'])/'tools/pack_reconstruction.py'),'track1','--episodes',str(view/'episodes'),'--sample',str(verify(c['sample'])),'--commit',c['commit_url'],'--out',str(candidate)]
    run=run_logged(argv,out/'stock_process',timeout=c['pack_timeout_s'],max_output_bytes=16*1024*1024,cwd=c['kit'])
    need(run['status']=='completed' and run['cleanup']['reaped'], 'stock process failed or not reaped')
    checked=roundtrip(c,view,candidate)
    # Derive permitted decoded coordinates from actual accepted rows, not only a candidate hash.
    table=pd.read_parquet(candidate); count=0
    allowed={(e,f) for e,source,f in lineage['accepted']}
    for ep in c['episodes']:
        e=ep['episode']; old=r.npz(ep['dense'],DENSE)
        for eid,f in allowed:
            if eid!=e: continue
            for a,k0 in [(0,'pose'),(4,'object_translation')]:
                rowid=f't1_{e:06d}_{f:06d}_{a}_000000'; value=table.loc[table.row_id==rowid,'z']
                need(len(value)==1, 'decoded allowed slot missing')
                count+=int(float(value.iloc[0])!=float(old[k0][f,2]))
    need(count>0, 'zero decoded numerical changes')
    baseline_diff=None
    if c['scope']=='official':
        oldtable=pd.read_parquet(r.path(c['baseline_candidate']))
        need(oldtable.columns.tolist()==table.columns.tolist() and oldtable.row_id.tolist()==table.row_id.tolist(),'old candidate exact row/order')
        oldxyz=oldtable[['x','y','z']].to_numpy(dtype=np.float64).copy(order='C');newxyz=table[['x','y','z']].to_numpy(dtype=np.float64).copy(order='C')
        need(np.isfinite(oldxyz).all(),'old candidate finite')
        permitted=np.zeros(oldxyz.shape,dtype=bool)
        for e,f in allowed:
            for a in (0,4):permitted[np.flatnonzero(table.row_id.to_numpy()==f't1_{e:06d}_{f:06d}_{a}_000000'),2]=True
        changed_bits=oldxyz.view(np.uint64)!=newxyz.view(np.uint64)
        need(not np.any(changed_bits & ~permitted),'old/new decoded difference outside permitted z cells')
        numeric=oldxyz!=newxyz
        need(np.count_nonzero(numeric)>0 and np.count_nonzero(np.any(numeric,axis=1))>0,'URL-only candidate forbidden')
        baseline_diff=dict(old_candidate=c['baseline_candidate'],exact_rows=True,changed_numeric_values=int(numeric.sum()),changed_bit_values=int(changed_bits.sum()),permitted_cells=int(permitted.sum()),all_other_values_bitwise_identical=True,URL_rows_changed=int((oldtable.code_commit_url!=table.code_commit_url).sum()))
        write_new(out/'baseline_numeric_difference.json',baseline_diff,True)
    r.finish(); verify_freeze(c['source_freeze'],c['source_tree_sha256']); verify_kit(c['kit'],json.loads(verify(c['kit_manifest']).read_text()))
    if c['scope']=='official': git_identity(Path(__file__).resolve().parents[1],c['source_commit'])
    result=dict(schema='t1_full_result_v1',scope=c['scope'],status='fixture_only_never_send' if c['scope']!='official' else 'full_package_awaiting_independent_acceptance',
                candidate=reference(candidate),sample=c['sample'],run_id=c['run_id'],execution_run_id=origin['run_id'],source_commit=c['source_commit'],commit_url=c['commit_url'],source_tree_sha256=c['source_tree_sha256'],policy_sha256=c['policy_sha256'],
                contract=reference(contract_path),validation_design=c['validation_design'],validation_protocol='prospective_all_A_proposals',original_sets=ORIGINAL32,accepted_frames=len(allowed),dense_changed_frames=len(allowed),decoded_changed_values=count,
                coverage=dict(episodes=len(plan['episodes']),frames=sum(len(e['required']) for e in plan['episodes']),source_runs=sum(len(e['sources']) for e in plan['episodes']),rows=len(rows)),
                provenance=reference(view/'provenance.json'),roundtrip=checked,baseline_numeric_difference=baseline_diff,stock_process=reference(out/'stock_process/finished.json'),model_calls=0,submission_ready=False)
    write_new(out/'full_result.json',result,True); return result
