"""Explicit fresh replay orchestration around the frozen C3 numerical engine.

No model, video or historical-production execution occurs on import. Production
resume remains in r12.py and rejects this mode. Every fresh run owns a new ledger.
"""
from pathlib import Path
from contextlib import contextmanager
import argparse, datetime, json, math, os, shutil, sys, time, uuid
import psutil
import r11_state as st
from r11_inputs import code_identity, numerical_hash, verify_code, audit_set
from r12_budget import validate_journal
from r12_supervision import execute_attempt, control_scope, read_attempt, SupervisionFailure
from .common import Paths, load, now, read_paths, ref, require, save_new, sha, token, digest
from .inputs import validate_roster, segments

SOURCE=Path(__file__).resolve().parents[1]
C2='483f8d61690d7df229c57dca43750b23afcfbf57'
C3='b433752cdd1828c34fa2ca9c328dd8b6ac2157e0'
CLEANUP={'exit_grace_s':5.,'term_wait_s':5.,'kill_wait_s':5.,'cleanup_total_s':20.,'telemetry_retry_count':2,'telemetry_retry_total_s':1.}
FIELDS=('counts','outcomes','written_bytes','cumulative_write_bytes')

def verify_fresh_root(root):
    root=Path(root).resolve();require(not root.is_relative_to(SOURCE),'fresh root cannot be inside the source checkout')
    man=load(root/'run_manifest.json')
    require(man.get('schema')=='r12_public_fresh_run_v1' and man.get('mode')=='fresh_reproduction','not an explicit public fresh root')
    require(man['root']==str(root) and man['historical_budget_parent'] is None,'fresh root identity/parent mismatch')
    require(man['scope'] in ('official','fixture'),'invalid fresh scope')
    verify_code(man);st.verify_ref(man['config'])
    return man

def reject_historical_ancestors(root):
    root=Path(root).resolve()
    require(not root.is_relative_to(SOURCE),'fresh output cannot overlap source')
    for parent in [root,*root.parents]:
        p=parent/'run_manifest.json'
        if p.is_file():
            prior=load(p)
            require(prior.get('schema') not in ('r11_run_v1','r12_run_v1','r12_public_fresh_run_v1'),'fresh init cannot nest in or reset an existing run')

def fresh_session(root,man,*,minutes):
    require(type(minutes) is int and 10<=minutes<=360,'explicit session10..360 minutes required')
    start=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)));mono=time.monotonic();duration=minutes*60
    # Separate monotonic deadlines survive wall-clock rollback within this boot.
    offsets={'new_episode_stop':max(1,duration-600),'candidate_stop':duration-90,'freeze_at':duration-45,'hard_stop':duration}
    return {'schema':'r12_public_fresh_session_v1','mode':'fresh_reproduction','scope':man['scope'],'session_id':uuid.uuid4().hex,
        'active_root':str(Path(root).resolve()),'created_at':now(),'start_at':start.isoformat(),'minutes':minutes,
        **{k:(start+datetime.timedelta(seconds=v)).isoformat() for k,v in offsets.items()},
        'monotonic_deadlines':{k:mono+v for k,v in offsets.items()},'boot_time':psutil.boot_time(),
        'cleanup':CLEANUP,'historical_budget_parent':None,'budget_semantics':'same fresh run total, baseline0; session renewal never resets current calls',
        'code':man['code'],'original_production_root_not_used':True}

def runtime_config(root,man=None):
    root=Path(root).resolve();man=man or verify_fresh_root(root);cfg=load(st.verify_ref(man['config']))
    latest=load(root/'session_latest.json');session=load(st.verify_ref(latest))
    require(session['schema']=='r12_public_fresh_session_v1' and session['active_root']==str(root) and session['code']==man['code'],'fresh session/root/code mismatch')
    require(session['historical_budget_parent'] is None and abs(session['boot_time']-psutil.boot_time())<1,'fresh session boot changed; explicit new-session required')
    require(time.monotonic()<session['monotonic_deadlines']['hard_stop'],'fresh session expired; explicit new-session required')
    return {**cfg,'fresh_session':session,'fresh_session_ref':latest,'budget_parent':None,
        'deadlines':{'new_episode_stop':session['new_episode_stop'],'candidate_stop':session['candidate_stop'],'candidate_freeze':session['freeze_at'],'hard_stop':session['hard_stop']}},session

class FreshLedger(st.Ledger):
    """Use the unchanged journal/reservation engine with an explicit fresh root."""
    def __init__(self,root,cfg):
        man=load(Path(root)/'run_manifest.json')
        require(man.get('schema')=='r12_public_fresh_run_v1' and cfg.get('budget_parent') is None and not cfg.get('execution_session'),'fresh ledger cannot enter production parent/registry path')
        require(cfg['budget_run_root']==str(Path(root).resolve()) and cfg['fresh_session']['active_root']==str(Path(root).resolve()),'fresh ledger/session root differs')
        super().__init__(root,cfg)
    def check_time(self,model=False,new_episode=False):
        session=self.cfg['fresh_session'];instant=time.monotonic()
        if instant>=session['monotonic_deadlines']['hard_stop']:raise st.BudgetStop('fresh_monotonic_hard_stop')
        if model and instant>=session['monotonic_deadlines']['candidate_stop']:raise st.BudgetStop('fresh_monotonic_candidate_stop')
        if new_episode and instant>=session['monotonic_deadlines']['new_episode_stop']:raise st.BudgetStop('fresh_new_episode_stop')
        if model and self.state_path.exists() and load(self.state_path).get('stop_reason'):raise st.BudgetStop('fresh_latched_resource_stop')

def validate_budget(root,cfg):
    ledger=FreshLedger(root,cfg);state=ledger.assert_no_unknown()
    require(state['baseline']=={'source':None},'fresh baseline must remain empty, no imported history')
    result=validate_journal(state,ledger.journal,{})
    require(all(state['accounting']['baseline'][k]=={} for k in FIELDS),'fresh baseline nonzero')
    return state,result

def assert_closed(root,exclude=None,owner_attempt=None):
    root=Path(root)
    for path in root.rglob('*.started.json'):
        if exclude is not None and path.resolve()==Path(exclude).resolve():continue
        value=read_attempt(path);require(value.get('supervisor_outcome')=='passed','unresolved fresh controller/worker: '+str(path))
    for path in root.rglob('owner_latest.json'):
        owner=load(path)
        if owner['attempt']==owner_attempt:continue
        require((path.parent/'attempts'/f'{owner["attempt"]}_end.json').exists(),'unclosed fresh owner')

def initialize(paths,preparation,roster_path,assets_path,*,minutes=150,scope='official'):
    require(scope in ('official','fixture'),'invalid scope')
    index=load(preparation);require(index['schema']=='r12_public_input_preparation_v1' and index['mode']=='pts' and index['scope']==scope,'fresh init requires a full PTS preparation index of matching scope')
    roster=load(roster_path);rows=validate_roster(roster,scope=scope);results=index['results']
    require(sorted(x['episode'] for x in results)==[r['episode'] for r in rows],'preparation must cover every roster episode exactly once')
    require(all(x['status']=='passed' and x['scope']==scope for x in results),'fresh init requires passed PTS preparation')
    code=code_identity();root=paths.output('fresh_runs/'+token());reject_historical_ancestors(root);root.mkdir(parents=True,exist_ok=False)
    # Fresh config contains only public numeric policy and explicit local assets,
    # never C1/C2 budget parents, private SPEC/adoption or old absolute deadlines.
    old=load(SOURCE/'config_r11_20261017a1.json');manifest=load(assets_path);assets={}
    for a in manifest['assets']:
        if a['key']=='kit':continue
        path=paths.asset(a['key'],a['logical_path'])
        if a['key']=='sample' and a['key'] not in paths.overrides:path=paths.asset('kit','external/v2d_submission_kit')/a['member']
        if scope=='official':require(path.is_file() and sha(path)==a['sha256'],'required external asset differs: '+a['key'])
        assets[a['key']]={'path':str(path),'sha256':a['sha256']}
    assets['roster']=ref(roster_path)
    cfg={k:old[k] for k in ('method','numerical_contract','call_limits','storage_limits','stage_limits','global_new_file_limit','monitor_interval_s')}
    cfg.update(root=str(paths.workspace),kit=str(paths.asset('kit','external/v2d_submission_kit')),mesh_site=str(SOURCE),assets=assets,
        budget_parent=None,budget_run_root=str(root),scope=scope,date=now()[:10],global_scan_interval_s=old['global_scan_interval_s'])
    # Legacy numerical workers use cfg.mesh_site to add mesh dependencies. The
    # child receives explicit extra sites via PYTHONPATH instead of private paths.
    cfg['python_extra_sites']=[]
    if scope=='official':require(numerical_hash(cfg)==load(SOURCE/'manifests/recipe_c2.json')['historical_numeric_hash'],'fresh numerical recipe signature changed')
    cfgref=save_new(root/'fresh_config.json',cfg)
    man={'schema':'r12_public_fresh_run_v1','mode':'fresh_reproduction','scope':scope,'root':str(root),'created_at':now(),
        'code':code,'config':cfgref,'historical_budget_parent':None,'recipe_by_episode':{str(r['episode']):C2 if r['episode']<23 else C3 for r in rows},
        'numerical_hash':numerical_hash(cfg),'audit':audit_set(roster) if scope=='official' else {'pairs':[],'full_mesh_pairs':[]},
        'preparation_index':ref(preparation),'actual_inference_performed_at_init':False,'build_commit':code['commit']}
    save_new(root/'run_manifest.json',man)
    session=fresh_session(root,man,minutes=minutes);sref=save_new(root/'sessions'/f'{session["session_id"]}.json',session);st.atomic_json(root/'session_latest.json',sref)
    runtime,_=runtime_config(root,man);ledger=FreshLedger(root,runtime);writer=st.Writer(ledger)
    by_id={x['episode']:x for x in results}
    for row in rows:
        item=by_id[row['episode']];source_ep=load(st.verify_ref(item['episode_input']));pts=load(st.verify_ref(item['whole_pts']))
        require(source_ep['scope']==scope and source_ep['whole_pts_verified'] and source_ep['required']==row['required_original_frames'],'prepared episode identity/frame conflict')
        require(source_ep['whole_pts']==item['whole_pts'] and pts.get('whole_pts_verified') is True and pts.get('status')=='passed','preparation episode/PTS refs or validation disagree')
        identity={k:row[k] for k in ('source_repository','revision','repository_path','declared_size_bytes','expected_sha256')}
        require(source_ep['source_identity']==identity and source_ep['video']['sha256']==row['expected_sha256'],'prepared source/video identity conflict')
        st.verify_ref(source_ep['video'])
        codehashes={'public_inputs.py':sha(SOURCE/'v2d_public/inputs.py'),'fresh.py':sha(Path(__file__))}
        bridge={**pts,'schema':'r12_public_fresh_pts_bridge_v1','identity':identity,'code_hashes':codehashes,'source_preparation_ref':item['whole_pts'],'bridge_created_at':now()}
        pr=writer.json(root/'input_pts'/f'episode_{row["episode"]:06d}.json',bridge,'small',True)
        ep={**source_ep,'schema':'r12_public_fresh_episode_v1','whole_pts':pr,'checker_hashes':codehashes,
            'recipe_commit':man['recipe_by_episode'][str(row['episode'])],'actual_prediction_producer_commit':None}
        writer.json(root/'inputs'/f'episode_{row["episode"]:06d}.json',ep,'small',True)
    state,_=validate_budget(root,runtime)
    require(not state['counts'] and not state['episode_counts'] and not ledger.journal.exists(),'fresh init must not invent calls')
    return {'root':str(root),'scope':scope,'session':sref,'baseline':state['baseline'],'current_counts':state['counts'],'prediction_producer_commit':None,'model_calls':0}

def renew(root,minutes):
    root=Path(root).resolve();man=verify_fresh_root(root);assert_closed(root)
    # Expired sessions are allowed only for this explicit renewal operation.
    cfg=load(st.verify_ref(man['config']));previous=load(st.verify_ref(load(root/'session_latest.json')))
    checkcfg={**cfg,'fresh_session':previous};validate_budget(root,checkcfg)
    session=fresh_session(root,man,minutes=minutes);r=save_new(root/'sessions'/f'{session["session_id"]}.json',session);st.atomic_json(root/'session_latest.json',r)
    return {'root':str(root),'session':r,'budget_reset':False}

def resource_gate(root,cfg):
    used=sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file());reserve=64*1024**2
    require(used<cfg['global_new_file_limit']-reserve and shutil.disk_usage(root).free>reserve+128*1024**2,'fresh resource/receipt headroom exhausted')
    ledger=FreshLedger(root,cfg)
    with ledger.edit() as state:state['global_observation']={'global_bytes':used,'local_accounted_bytes':sum(state['written_bytes'].values()),'as_of':now()}
    return used

def observe_artifact(root,episode,stage):
    latest=Path(root)/'episode_runs'/f'episode_{episode:06d}'/f'{stage}_latest.json'
    if not latest.exists():return {'status':'absent','contract':'unverified','refs':[]}
    r=load(latest);value=load(st.verify_ref(r))
    return {'status':'completed' if value['status']=='completed' else 'failed','contract':'unverified','refs':[ref(latest),ref(r['path'])]}

def validate_stage(root,episode,stage,man,cfg):
    er=Path(root)/'episode_runs'/f'episode_{episode:06d}';record=load(st.verify_ref(load(er/f'{stage}_latest.json')))
    require(record['status']=='completed' and record['stage']==stage and record['episode']==episode and record['producer']==man['code']['commit'] and record['scope']==man['scope'],'fresh stage artifact identity failed')
    ep=load(Path(root)/'inputs'/f'episode_{episode:06d}.json');details=record['details']
    ledger=FreshLedger(root,cfg);validate_budget(root,cfg)
    if stage=='A':
        require(set(st.FrameStore(er,st.Writer(ledger)).records())==set(ep['required']),'fresh A coverage differs')
    elif stage=='B':
        for key in ('plan','identity','mesh'):st.verify_ref(details[key])
        require(details['rig_checked']==ep['required'],'fresh B coverage differs')
    elif stage=='C':
        for r in details['inputs'].values():
            if isinstance(r,dict) and 'path' in r and 'sha256' in r:st.verify_ref(r)
        require(details['frames']==ep['required'],'fresh C coverage differs')
    else:require(man['scope']=='fixture' and details.get('fixture') is True,'fixture cannot be used as inference')
    return record

def execute(root,episodes,*,allow_models=False,fixture=False,extra_sites=()):
    root=Path(root).resolve();man=verify_fresh_root(root);cfg,session=runtime_config(root,man)
    require(not (fixture and allow_models),'fixture and model permission cannot be combined')
    require((fixture and man['scope']=='fixture') or (not fixture and man['scope']=='official' and allow_models),'explicit --execute-models and official preparation required for inference')
    wanted=sorted(set(episodes));available={int(p.stem.split('_')[1]) for p in (root/'inputs').glob('episode_*.json')};require(set(wanted)<=available and wanted,'unknown/empty episodes')
    assert_closed(root);validate_budget(root,cfg);prefix=root/'supervision'/('fresh_controller_'+uuid.uuid4().hex)
    owner=st.Owner(root/'controller_owner','fresh_run',None)
    with control_scope(owner,prefix,metadata={'mode':'fresh_reproduction','scope':man['scope'],'code':man['code'],'session_ref':cfg['fresh_session_ref']}) as current:
        for episode in wanted:
            for stage in (('fixture',) if fixture else ('A','B','C')):
                if observe_artifact(root,episode,stage)['status']=='completed':validate_stage(root,episode,stage,man,cfg);continue
                validate_budget(root,cfg);resource_gate(root,cfg)
                lease={'schema':'r12_public_fresh_worker_lease_v1','root':str(root),'session_ref':cfg['fresh_session_ref'],'code':man['code'],
                    'controller_pid':os.getpid(),'controller_create_time':psutil.Process().create_time(),'controller_attempt':current.attempt,
                    'episode':episode,'stage':stage,'scope':man['scope'],'model_permission':allow_models and not fixture}
                lease_ref=save_new(root/'leases'/f'{uuid.uuid4().hex}.json',lease)
                cmd=[sys.executable,'-B',str(SOURCE/'tools/fresh_worker.py'),'--root',str(root),'--episode',str(episode),'--stage',stage,'--lease',lease_ref['path'],'--lease-hash',lease_ref['sha256']]
                env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false');env['PYTHONPATH']=os.pathsep.join(str(Path(p).resolve()) for p in extra_sites)
                limits=cfg['stage_limits']['C' if stage=='fixture' else stage]
                def preflight():
                    # The new supervision started record already exists here.
                    # Global prior-attempt checks ran before acquiring this sole
                    # controller owner; execute_attempt excludes its own attempt.
                    verify_code(man);validate_budget(root,cfg)
                last_scan=[time.monotonic()]
                def resources(elapsed,rss):
                    FreshLedger(root,cfg).check_time(model=True)
                    if time.monotonic()-last_scan[0]>=cfg['global_scan_interval_s']:
                        resource_gate(root,cfg);last_scan[0]=time.monotonic()
                result=execute_attempt(directory=root/'supervision'/f'episode_{episode:06d}',command=cmd,cwd=SOURCE,session=session,code=man['code'],config_ref=man['config'],
                    stage=stage,episode=episode,owner_id=current.attempt,parent_attempt=current.attempt,parent_refs=[lease_ref,cfg['fresh_session_ref']],env=env,
                    monitor_interval_s=.05,wall_s=min(limits['wall_s'],max(.001,session['monotonic_deadlines']['candidate_stop']-time.monotonic())),rss_bytes=limits['rss_bytes'],
                    preflight=preflight,artifact=lambda:observe_artifact(root,episode,stage),validate_artifact=lambda value:validate_stage(root,episode,stage,man,cfg),
                    resource_check=resources,postflight=lambda:verify_code(man),stop_types=(st.BudgetStop,))
                require(result['supervisor_outcome']=='passed','fresh supervision did not pass')
    return {'root':str(root),'scope':man['scope'],'episodes':wanted,'mode':'fixture' if fixture else 'actual_fresh_inference','producer_commit':None if fixture else man['code']['commit'],'model_execution_in_fixture':False}

def export(root):
    root=Path(root).resolve();man=verify_fresh_root(root);require(man['scope']=='official','fixture cannot export an official view')
    cfg,session=runtime_config(root,man);assert_closed(root);validate_budget(root,cfg)
    for episode in range(30):
        for stage in ('A','B','C'):validate_stage(root,episode,stage,man,cfg)
    view=root/'exports'/token();view.mkdir(parents=True,exist_ok=False);writer=st.Writer(FreshLedger(root,cfg));entries=[]
    for episode in range(30):
        er=root/'episode_runs'/f'episode_{episode:06d}';record=load(st.verify_ref(load(er/'C_latest.json')));ep=load(root/'inputs'/f'episode_{episode:06d}.json');inputs=record['details']['inputs'];artifacts={}
        for key,logical in [('npz',f'episodes/episode_{episode:06d}.npz'),('mask',f'masks/episode_{episode:06d}.npz'),('obj',f'episodes/episode_{episode:06d}_object.obj')]:
            r=inputs[key];data=st.verify_ref(r).read_bytes();rr=writer.bytes(view/logical,data,'C',True);require(rr['sha256']==r['sha256'],'fresh export bytes changed');artifacts[key]={'logical_path':logical,'sha256':rr['sha256']}
        entries.append({'episode':episode,'prediction_producer_commit':man['code']['commit'],'producer_tree':man['code']['tree'],'recipe_commit':man['recipe_by_episode'][str(episode)],'imported':False,'input_hash':ep['video']['sha256'],'required_digest':digest(ep['required']),'importer_commit':None,'pack_producer_commit':man['code']['commit'],**artifacts})
    provenance={'schema':'r12_public_provenance_v1','mode':'fresh_reproduction','build_commit':man['code']['commit'],'episodes':entries,'historical_artifact_identity_claimed':False}
    save_new(view/'provenance.json',provenance)
    return {'view':str(view),'provenance':ref(view/'provenance.json'),'stock_writer_executed':False,'published':False}

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('assets','inputs','pack','environment'):
        command=argv.pop(0)
        if command=='assets':from .assets import main as delegated
        elif command=='inputs':from .inputs import main as delegated
        elif command=='pack':from .packing import main as delegated
        else:from .environment import main as delegated
        return delegated(argv)
    p=argparse.ArgumentParser(description='Fresh replay: prepare/import bridges, explicit independent session/ledger, C3-supervised inference and stock export. No production resume.')
    sub=p.add_subparsers(dest='mode',required=True)
    init=sub.add_parser('init');init.add_argument('--paths',type=Path,required=True);init.add_argument('--workspace');init.add_argument('--preparation',type=Path,required=True);init.add_argument('--roster',type=Path,default=SOURCE/'manifests/input_roster.json');init.add_argument('--assets',type=Path,default=SOURCE/'manifests/external_assets.json');init.add_argument('--minutes',type=int,default=150);init.add_argument('--scope',choices=['official','fixture'],default='official')
    renewp=sub.add_parser('new-session');renewp.add_argument('--root',type=Path,required=True);renewp.add_argument('--minutes',type=int,default=150)
    run=sub.add_parser('run');run.add_argument('--root',type=Path,required=True);run.add_argument('--episode',type=int,action='append',required=True);run.add_argument('--execute-models',action='store_true');run.add_argument('--fixture',action='store_true');run.add_argument('--extra-site',action='append',default=[])
    for command in ('status','export'):
        parser=sub.add_parser(command);parser.add_argument('--root',type=Path,required=True)
    a=p.parse_args(argv)
    if a.mode=='init':paths,_=read_paths(a.paths,a.workspace);result=initialize(paths,a.preparation,a.roster,a.assets,minutes=a.minutes,scope=a.scope)
    elif a.mode=='new-session':result=renew(a.root,a.minutes)
    elif a.mode=='run':result=execute(a.root,a.episode,allow_models=a.execute_models,fixture=a.fixture,extra_sites=a.extra_site)
    elif a.mode=='export':result=export(a.root)
    else:
        man=verify_fresh_root(a.root);cfg,_=runtime_config(a.root,man);state,validation=validate_budget(a.root,cfg);result={'root':str(a.root.resolve()),'scope':man['scope'],'session':cfg['fresh_session_ref'],'budget':state,'current_closed_calls':validation['calls'],'historical_budget_parent':None,'production_root_used':False}
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0
