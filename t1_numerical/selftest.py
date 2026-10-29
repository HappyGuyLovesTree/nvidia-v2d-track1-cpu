"""End-to-end original-format synthetic replay, relocation and rejection checks."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from fixture_data import build
from control import reference,write_new
from geometry import need
from d_common import bit_equal

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out).resolve();need(not out.exists(),'selftest output exists');out.mkdir()
    source=Path(__file__).resolve().parent;dataset=build(out/'dataset_original');checks=[]
    def check(label,condition):need(bool(condition),'selftest failed '+label);checks.append(label)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',T1_FIXTURE_IMPORT_GUARD='1')
    env['PYTHONPATH']=str(source/'fixture_guard')+os.pathsep+env.get('PYTHONPATH','')
    def run(code,data,result,mode='run',succeed=True):
        command=[sys.executable,'-B',str(code/'portable.py'),mode,'--dataset',str(data),'--out',str(result),'--fixture','--rig-cap','64']
        cp=subprocess.run(command,env=env,capture_output=True,text=True)
        write_new(out/(result.name+'_CLI.json'),dict(argv=command,exit=cp.returncode,stdout=cp.stdout,stderr=cp.stderr))
        check(result.name+'_exit',cp.returncode==(0 if succeed else 1))
        return json.loads((result/'numerical_result.json').read_text()) if succeed else None
    a=run(source,dataset,out/'run_original')
    check('full_R0_to_P_to_statistics_to_geometry',a['counts']['proposals']==2 and a['geometric_accepts']==1 and a['counts']['allowed_frames']==30)
    check('fixture_never_real_models',a['fixture'] and a['real_model_loads']==a['real_rig']==0 and a['synthetic_rig']==8)
    check('no_quality_or_submission_pass',not a['quality_pass'] and not a['submission_eligible'])
    code=out/'relocated code';shutil.copytree(source,code,ignore=shutil.ignore_patterns('__pycache__'))
    moved=out/'relocated data';shutil.copytree(dataset.parent,moved)
    config=json.loads((moved/'dataset.json').read_text());config['prefix_map']=[[str(dataset.parent/'r0'),'r0']]
    (moved/'dataset.json').write_text(json.dumps(config))
    b=run(code,moved/'dataset.json',out/'run_relocated')
    check('relocated_code_source_tree_same',a['source_tree_sha256']==b['source_tree_sha256'])
    check('relocated_counts_equal',a['counts']==b['counts'] and a['geometric_accepts']==b['geometric_accepts'] and a['first_reason_counts']==b['first_reason_counts'])
    for x,y in zip(a['exports'],b['exports']):
        with np.load(x['dense']['path'],allow_pickle=False) as za,np.load(y['dense']['path'],allow_pickle=False) as zb:
            check('relocated_six_arrays_bit_equal_'+str(x['episode']),set(za.files)==set(zb.files) and all(bit_equal(za[k],zb[k]) for k in za.files))
        original=dataset.parent/'r0/episodes'/f"episode_{x['episode']:06d}.npz"
        with np.load(original,allow_pickle=False) as zo,np.load(x['dense']['path'],allow_pickle=False) as zn:
            for key in ('pose','object_translation'):
                check('unselected_NaN_payload_preserved_'+str(x['episode'])+'_'+key,bit_equal(zo[key][-3:],zn[key][-3:]))
        for key in ('mask','obj','mesh'):check('copied_'+str(x['episode'])+'_'+key,x['copied'][key]['sha256']==y['copied'][key]['sha256'])
    ba=json.loads(Path(a['geometry_bundle']['path']).read_text());bb=json.loads(Path(b['geometry_bundle']['path']).read_text())
    check('actual_D_bundle_schema',ba['schema']=='t1_persisted_geometry_v1' and len(ba['rows'])==2)
    for x,y in zip(ba['rows'],bb['rows']):
        check('relocated_geometry_decision_'+str(x['frame']),x['gate']==y['gate'] and x['adapter']==y['adapter'] and x['proposal_id']==y['proposal_id'])
        for key in ('old_params','candidate_params'):
            with np.load(x[key]['path'],allow_pickle=False) as z:
                check('exact_param_schema_'+str(x['frame'])+'_'+key,set(z.files)=={'q','shape','scales','t','R','scale','K'} and z['q'].shape==(136,) and z['shape'].shape==(45,) and z['scales'].shape==(68,))
        for key in ('old_call','new_call'):
            call=json.loads(Path(x[key]['path']).read_text())
            check('canonical_call_'+str(x['frame'])+'_'+key,call['schema']=='t1_persisted_rig_call_v1' and call['fixture'] and call['coordinate']=='mhr_native' and call['start']['call_id']==call['end']['call_id'])
        for key in ('old_rig','new_rig'):
            with np.load(x[key]['path'],allow_pickle=False) as z:check('full_native_rig_'+str(x['frame'])+'_'+key,set(z.files)=={'vertices','joints'} and z['vertices'].shape==(18439,3) and z['joints'].shape==(127,3))
        check('real_mesh_binding_'+str(x['frame']),set(x['mesh_binding'])>={'obj_ref','mesh_npz_ref','vertices_identity','faces_identity','old_transform_params','candidate_transform_params'})
    wrong=out/'wrong_count.json';cfg=json.loads(dataset.read_text());cfg['episodes']=[dict(e,**{k:str(dataset.parent/e[k]) for k in ('run_root','dense','mask','obj','mesh')}) for e in cfg['episodes']];cfg['expected']['required_frames']=39;wrong.write_text(json.dumps(cfg))
    run(source,wrong,out/'reject_count',mode='plan',succeed=False)
    check('wrong_count_pre_model',not(out/'reject_count/calls.jsonl').exists())
    bad=out/'bad_mesh';shutil.copytree(dataset.parent,bad);cfg=json.loads((bad/'dataset.json').read_text());cfg['prefix_map']=[[str(dataset.parent/'r0'),'r0']];(bad/'dataset.json').write_text(json.dumps(cfg))
    obj=bad/'r0/episodes/episode_000000_object.obj';obj.write_text(obj.read_text().replace('0.10000000000000001','0.2'))
    run(source,bad/'dataset.json',out/'reject_mesh',succeed=False)
    check('mesh_conflict_pre_model',not(out/'reject_mesh/calls.jsonl').exists())
    badref=out/'bad_parent';shutil.copytree(dataset.parent,badref);cfg=json.loads((badref/'dataset.json').read_text());cfg['prefix_map']=[[str(dataset.parent/'r0'),'r0']];(badref/'dataset.json').write_text(json.dumps(cfg))
    frame=badref/'r0/episode_runs/episode_000000/frames/000000.json';frame.write_bytes(frame.read_bytes()+b' ')
    run(source,badref/'dataset.json',out/'reject_parent_hash',mode='plan',succeed=False)
    check('parent_hash_conflict_pre_model',not(out/'reject_parent_hash/calls.jsonl').exists())
    empty=out/'no_scope.json';cfg=json.loads(dataset.read_text());cfg['episodes']=[dict(cfg['episodes'][1],**{k:str(dataset.parent/cfg['episodes'][1][k]) for k in ('run_root','dense','mask','obj','mesh')})]
    cfg['expected']=dict(episodes=1,required_frames=8,source_runs=1,allowed_segments=0,allowed_frames=0,proposals=0)
    empty.write_text(json.dumps(cfg));zero=run(source,empty,out/'zero_proposals')
    check('zero_proposals_no_private_answer_fallback',zero['counts']['proposals']==0 and zero['status']=='no_proposals_stop')
    check('zero_proposals_no_models_or_export',zero['real_rig']==0 and not(out/'zero_proposals/calls.jsonl').exists() and not(out/'zero_proposals/episodes').exists())
    check('selftest_process_no_real_runtime',not any(k.split('.')[0] in ('torch','onnxruntime','cv2','av') for k in sys.modules))
    result=dict(schema='t1_portable_selftest_v1',passed=len(checks),failed=0,checks=checks,real_model_calls=0,new_real_data_read=False,
        model_import_firewall=True,original_fixture_result=reference(out/'run_original/numerical_result.json'),relocated_fixture_result=reference(out/'run_relocated/numerical_result.json'),
        lineage_path_refs_expected_to_differ=True,real_MHR_execution_tested=False,real_dataset_replay_tested=False)
    write_new(out/'selftest_result.json',result);print(json.dumps({k:result[k] for k in ('passed','failed','real_model_calls','real_MHR_execution_tested')}))

if __name__=='__main__':main()
