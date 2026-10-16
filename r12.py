"""R012 production-resume CLI. No fresh mode, implicit retries or legacy recomputation."""
from pathlib import Path
from contextlib import contextmanager
import argparse, datetime, fcntl, json, os, shutil, sys, uuid
from common import load, sha
from tracking import digest
from r11_state import *
from r11_inputs import code_identity, verify_code, audit_set, numerical_hash, load_config
from r12_budget import signed_chain, validate_chain, validate_current, C2
from r12_session import *
from r12_import import legacy_entry, copy_checked, verify_layers, verify_frozen_input, export_layers
from r12_supervision import control_scope, read_attempt

P=Path(__file__).resolve().parent
DEFAULT_INDEX=P.parents[2]/'证据/T011.08/evidence_index_C2最终_20261017a1.json'

@contextmanager
def registry_lock(cfg):
    d=global_directory(cfg);d.mkdir(parents=True,exist_ok=True)
    with (d/'owner.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield d
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def initialize(root,chain_ref,handover_ref,authorization,diff_ref,minutes):
    root=Path(root).resolve();base=validate_chain(chain_ref);chain=load(chain_ref['path'])
    cfg,roster=load_config(chain['parent_config_ref']['path']);code=code_identity()
    require(Path(sys.executable).resolve()==Path('/Users/zhuanz/.venvs/openwebui/bin/python').resolve(),'use frozen Python')
    require(not root.exists(),'new root already exists; use new-session to preserve its charges')
    with registry_lock(cfg) as directory:
        require(not (directory/'registry.json').exists(),'root already registered; reuse existing root')
        man={'schema':'r12_run_v1','created_at':now(),'code':code,'config':chain['parent_config_ref'],
             'audit':audit_set(roster),'numeric_hash':numerical_hash(cfg),'kit_source_hashes':{str(p):sha(p) for p in Path(cfg['kit']).rglob('*.py')},
             'parent_bundle_ref':chain_ref,'parent_index_ref':chain['parent_index_ref'],'historical_handover_ref':handover_ref,
             'provenance_kind':'local_git','submission_eligible':False}
        session=make_session(root,man,chain_ref,handover_ref,authorization,diff_ref,minutes=minutes)
        root.mkdir(parents=True);sp=root/'sessions'/f"{session['session_id']}.json";atomic_json(sp,session,True)
        man['session_ref']=ref(sp);atomic_json(root/'run_manifest.json',man,True)
        atomic_json(directory/'registry.json',{'schema':'r12_single_production_root_v1','active_root':str(root),'parent_bundle_ref':chain_ref,'created_at':now()},True)
        effective=runtime_config(man,root);ledger=Ledger(root,effective)
        with ledger.edit():pass
        writer=Writer(ledger)
        resource_gate(root,effective,ledger)
        bootstrap=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and 'budget' not in p.relative_to(root).parts)
        ledger.reserve_bytes('small',bootstrap,write_bytes=bootstrap)
        writer.json(root/'bootstrap_accounting.json',{'bytes':bootstrap,'scope':'initial manifest/session; mutable ledger and locks counted in global DD scan'},exclusive=True)
        index=load(verify_ref(chain['parent_index_ref']));pipeline=load(verify_ref(index['frozen_files']['pipeline_index']))
        for row in pipeline['episodes']:
            roster_row=next(r for r in roster['episodes'] if r['episode']==row['episode'])
            verify_frozen_input(row['input'],roster_row,cfg['assets']['roster'])
            copy_checked(row['input'],root/'inputs'/f"episode_{row['episode']:06d}.json",writer,'small')
        return {'root':str(root),'manifest':ref(root/'run_manifest.json'),'session_ref':man['session_ref'],'production_release':False}

def renew(root,authorization,diff_ref,minutes,release_ref=None):
    man=load(root/'run_manifest.json');verify_code(man);oldcfg=load(verify_ref(man['config']))
    with registry_lock(oldcfg):
        validate_chain(man['parent_bundle_ref'])
        check_registry(global_directory(oldcfg),root,man['parent_bundle_ref'])
        assert_closed(root)
        session=make_session(root,man,man['parent_bundle_ref'],man['historical_handover_ref'],authorization,diff_ref,minutes=minutes,release_ref=release_ref)
        # Check persisted charges directly; an expired old session cannot authorize
        # new writes, but renewal never replaces its ledger or its baseline.
        probe={**oldcfg,'budget_parent':man['parent_bundle_ref']};validate_current(root,probe)
        path=root/'sessions'/f"{session['session_id']}.json";atomic_json(path,session,True)
        man['session_ref']=ref(path);atomic_json(root/'run_manifest.json',man)
        cfg=runtime_config(man,root);ledger=Ledger(root,cfg)
        # Conservatively charge the full new manifest write and new session.
        n=path.stat().st_size;ledger.reserve_bytes('small',n,write_bytes=n+(root/'run_manifest.json').stat().st_size)
        return {'session_ref':man['session_ref'],'counts_and_bytes':'continued_in_same_ledger'}

def context(root):
    man=load(root/'run_manifest.json');verify_code(man);cfg=runtime_config(man,root)
    return man,cfg,Ledger(root,cfg)

def scope(root,cfg,operation,evidence=None):
    prefix=root/'supervision'/f'controller_{operation}_{uuid.uuid4().hex}'
    return control_scope(GlobalOwner(root,cfg,prefix),prefix,metadata={'session_id':cfg['execution_session']['session_id'],'operation':operation},evidence=evidence)

def imports(root,cfg=None):
    out=[]
    pipeline=None
    if cfg:
        chain=load(verify_ref(cfg['budget_parent']));index=load(verify_ref(chain['parent_index_ref']))
        pipeline=load(verify_ref(index['frozen_files']['pipeline_index']))
    for p in sorted((root/'imported').glob('episode_*.json')):
        entry=load(p);require(entry['schema']=='r12_imported_completed_v1' and entry['status']=='imported_completed','invalid imported schema')
        require(entry['prediction_producer_commit']==C2,'import producer changed')
        if cfg:
            require(entry['parent_index_ref']==load(cfg['budget_parent']['path'])['parent_index_ref'],'import parent changed')
            row=next(r for r in load(cfg['assets']['roster']['path'])['episodes'] if r['episode']==entry['episode'])
            ep=verify_frozen_input(entry['episode_input_ref'],row,cfg['assets']['roster'])
            require(entry['required']==ep['required'] and entry['required_digest']==digest(ep['required']) and entry['sources_digest']==digest(ep['sources']),'import required/source changed')
            original=next(r for r in pipeline['episodes'] if r['episode']==entry['episode'])
            require(entry['episode_input_ref']==original['input'] and entry['validation_ref']==original['pack'] and entry['C_receipt_ref']==original['stages']['C']['receipt'],'import frozen provenance mismatch')
            pack=load(verify_ref(original['pack']))
            require(entry['source_refs']==pack['inputs'][str(entry['episode'])] and entry['producer_tree']=='488470c7217dedb23e72abe70e896ca4b3e078d8','import source/producer mismatch')
        verify_layers(entry['source_refs'],entry['required']);verify_layers(entry['export_refs'],entry['required'])
        require(all(entry['source_refs'][k]['sha256']==entry['export_refs'][k]['sha256'] for k in ('npz','obj','mask','mesh')),'import copy changed')
        out.append(entry)
    return out

def do_import(root):
    man,cfg,ledger=context(root);writer=Writer(ledger)
    with scope(root,cfg,'import'):
        resource_gate(root,cfg,ledger)
        index=load(verify_ref(man['parent_index_ref']));pipeline=load(verify_ref(index['frozen_files']['pipeline_index']))
        cfg0=load(verify_ref(man['config']));copied=0
        for row in pipeline['episodes']:
            if row['episode']>22:continue
            path=root/'imported'/f"episode_{row['episode']:06d}.json"
            if path.exists():continue
            entry=legacy_entry(row,cfg0,man['parent_index_ref'],index['frozen_files']['manifest'])
            e=entry['episode'];locations={'npz':root/'legacy/episodes'/f'episode_{e:06d}.npz','obj':root/'legacy/episodes'/f'episode_{e:06d}_object.obj',
                                       'mask':root/'legacy/masks'/f'episode_{e:06d}.npz','mesh':root/'legacy/meshes'/f'episode_{e:06d}.npz'}
            entry['export_refs']={k:copy_checked(entry['source_refs'][k],p,writer) for k,p in locations.items()}
            entry['importer_code']=man['code'];entry['session_id']=cfg['execution_session']['session_id'];entry['copied_bytes']=sum(r['bytes'] for r in entry['export_refs'].values());copied+=entry['copied_bytes']
            writer.json(path,entry,'small',True)
        entries=imports(root,cfg);require([e['episode'] for e in entries]==list(range(23)),'import not complete23')
        receipt={'schema':'r12_import_summary_v1','created_at':now(),'entries':[ref(root/'imported'/f'episode_{e:06d}.json') for e in range(23)],
                 'completed_episodes':list(range(23)),'frame_count':sum(len(e['required']) for e in entries),'source_count':58,
                 'copied_bytes_this_command':copied,'copied_artifact_bytes_total':sum(e['copied_bytes'] for e in entries),
                 'first240_imported':len(entries[16]['required'])==240,'model_loads':0,'model_forwards':0,'video_decodes':0,
                 'historical_C22_exit_reap':'unverified','importer_code':man['code']}
        p=root/'import_summaries'/f"{uuid.uuid4().hex}.json";writer.json(p,receipt,'small',True);writer.json(root/'import_latest.json',ref(p))
        return {'summary':ref(p),'copied_bytes':copied,'completed':23}

def new_entries(root,man,cfg,active_owner_id=None):
    from r11 import stage_receipt, validate_completed
    result=[]
    for e in range(23,30):
        ep=load(root/'inputs'/f'episode_{e:06d}.json');er=root/'episode_runs'/f'episode_{e:06d}'
        if not validate_completed(root,ep,cfg):continue
        # Numeric commit alone is insufficient. All stage and supervisor/controller
        # results must close before this new completion enters an export view.
        for stage in ('A','B','C'):
            receipt=stage_receipt(er,stage);require(receipt and receipt['status']=='completed' and receipt['producer']==man['code']['commit'],'new episode stage not completed')
            matches=[read_attempt(p) for p in (root/'supervision').glob(f'{stage}_{e}_*.started.json')]
            require(matches and all(r['supervisor_outcome']=='passed' for r in matches),'new episode supervisor not passed')
            for attempt in matches:
                directory=global_directory(cfg);claim=load(directory/'attempts'/f"{attempt['owner_id']}.json")
                require(claim['active_root']==str(root) and claim['session_id']==attempt['session_id'],'attempt/controller binding')
                if attempt['owner_id']==active_owner_id:
                    require(claim['pid']==os.getpid() and abs(psutil.Process().create_time()-claim['pid_started'])<.001,'active controller mismatch')
                else:
                    end=load(directory/'attempts'/f"{claim['attempt']}_end.json")
                    require(end['outcome']=='completed' and read_attempt(Path(claim['controller_prefix']).with_suffix('.started.json'))['supervisor_outcome']=='passed','controller finalization not passed')
        c=stage_receipt(er,'C');entry=c['details']['inputs'];verify_layers(entry,ep['required'])
        result.append({'episode':e,'required':ep['required'],'source_refs':entry,'validation_ref':c['details']['pack'],
                       'prediction_producer_commit':man['code']['commit'],'producer_tree':man['code']['tree']})
    return result

def release_gate(root,man,cfg,episodes):
    session=cfg['execution_session'];r=session.get('production_release_ref');require(r,'independent production release missing')
    release=load(verify_ref(r));require(release['schema']=='r12_production_release_v1','production release schema')
    require(release['code_commit']==man['code']['commit'] and release['active_root']==str(root),'release code/root mismatch')
    require(release['sup_acceptance']=='passed' and release['ses_budget_import_acceptance']=='passed','independent gates not passed')
    require(set(episodes)<=set(release['allowed_episodes'])<=set(range(23,30)),'episodes not released')
    for evidence_ref in release['independent_evidence_refs']:verify_ref(evidence_ref)
    require(bool(release['independent_evidence_refs']),'independent evidence missing')
    summary=load(verify_ref(load(root/'import_latest.json')));require(summary['completed_episodes']==list(range(23)) and summary['first240_imported'],'legacy23/first240 gate')
    return release

def projection(root,cfg,completed):
    # Remaining work uses committed stage journals/receipts, so already spent
    # closed calls are not projected a second time. Loads needed for a resumed
    # partially completed stage remain charged as one additional actual load.
    from r11 import stage_receipt
    state,_=validate_current(root,cfg);need={k:0 for k in cfg['call_limits']}
    for e in range(23,30):
        if e in completed:continue
        ep=load(root/'inputs'/f'episode_{e:06d}.json');n=len(ep['required']);er=root/'episode_runs'/f'episode_{e:06d}'
        for stage,kinds in (('A',('detector','hmr')),('B',('rig',))):
            rr=stage_receipt(er,stage)
            if rr and rr['status']=='completed':continue
            for k in kinds:need[k]+=max(0,(2*n if k=='rig' else n)-state['episode_counts'].get(str(e)+':'+k,0))
            for k in (('dnn_load','ort_session_load') if stage=='A' else ('rig_load',)):need[k]+=1
    for k,n in need.items():require(state['counts'].get(k,0)+n<=cfg['call_limits'][k],'remaining projection exceeds cap: '+k)
    return {'needed':need,'projected_total':{k:state['counts'].get(k,0)+n for k,n in need.items()},'cap':cfg['call_limits']}

def resource_gate(root,cfg,ledger):
    from r11 import global_new_bytes
    used=global_new_bytes(cfg);reserve=67108864
    require(used<cfg['global_new_file_limit']-reserve,'global original-baseline headroom exhausted')
    require(shutil.disk_usage(root).free>=reserve+134217728,'insufficient physical disk/receipt headroom')
    with ledger.edit() as s:s['global_observation']={'global_bytes':used,'local_accounted_bytes':sum(s['written_bytes'].values()),'as_of':now()}
    return used

def run(root,episodes):
    from r11 import supervise,stage_receipt,validate_completed
    man,cfg,ledger=context(root);release_gate(root,man,cfg,episodes)
    with scope(root,cfg,'run') as owner:
        old=imports(root,cfg);require(len(old)==23 and old[16]['episode']==16,'legacy23 not verified')
        if any(e!=28 for e in episodes):require(any(e['episode']==28 for e in new_entries(root,man,cfg)),'episode28 must pass before remaining6')
        for e in episodes:
            require(e in range(23,30),'legacy recomputation prohibited');ledger.check_time(model=True)
            if any(v['episode']==e for v in new_entries(root,man,cfg,owner.attempt)):continue
            projection(root,cfg,{v['episode'] for v in new_entries(root,man,cfg,owner.attempt)});resource_gate(root,cfg,ledger)
            ep=load(root/'inputs'/f'episode_{e:06d}.json');er=root/'episode_runs'/f'episode_{e:06d}'
            if not any(ledger.snapshot()['episode_counts'].get(str(e)+':'+k,0) for k in ('dnn_load','ort_session_load','rig_load')):ledger.check_time(model=True,new_episode=True)
            for stage in ('A','B','C'):
                rr=stage_receipt(er,stage)
                if rr and rr['status']=='completed':continue
                supervise(root,cfg,man,stage,e,owner_id=owner.attempt,parent_attempt=owner.record['parent_attempt'])
                validate_current(root,cfg)
    return status(root,man,cfg)

def aggregate_worker(root):
    from r11_pack import pack_local
    man,cfg,ledger=context(root);child_bridge(root,cfg,stage='aggregate');writer=Writer(ledger)
    old=imports(root,cfg);new=new_entries(root,man,cfg);entries=old+new
    require({e['episode'] for e in old}==set(range(23)),'legacy23 missing')
    require({e['episode'] for e in new}==set(range(23,30)),'new7 missing; full30 required')
    require(sum(len(e['required']) for e in entries)==9877,'full required count')
    directory=root/'export_views'/uuid.uuid4().hex
    view,coverage,inputs=export_layers(directory,entries,writer,man['code'])
    result=pack_local(directory,cfg,man,coverage,inputs,writer,'full30')
    require(result['rows']==740780 and result['coverage']=='full','full30 pack failed')
    writer.json(root/'aggregate_latest.json',ref(directory/'packs/full30/validation.json'))
    writer.json(root/'export_view_latest.json',ref(directory/'view.json'))
    return {'pack':load(root/'aggregate_latest.json'),'view':load(root/'export_view_latest.json')}

def aggregate(root):
    from r11 import supervise
    man,cfg,ledger=context(root);release_gate(root,man,cfg,list(range(23,30)))
    with scope(root,cfg,'aggregate') as owner:
        resource_gate(root,cfg,ledger)
        return supervise(root,cfg,man,'aggregate',None,owner_id=owner.attempt,parent_attempt=owner.record['parent_attempt'])

def status(root,man=None,cfg=None):
    man=man or load(root/'run_manifest.json');cfg=cfg or runtime_config(man,root)
    state,result=validate_current(root,cfg)
    old=imports(root,cfg);new=new_entries(root,man,cfg)
    return {'schema':'r12_status_v1','created_at':now(),'active_root':str(root),'session_ref':man['session_ref'],'code':man['code'],
            'completed_legacy':[e['episode'] for e in old],'completed_new':[e['episode'] for e in new],
            'remaining':[e for e in range(23,30) if e not in {x['episode'] for x in new}],
            'budget':{k:state[k] for k in ('counts','outcomes','written_bytes','cumulative_write_bytes','baseline')},
            'projection':projection(root,cfg,{e['episode'] for e in new}),'historical_C22_exit_reap':'unverified','submission_eligible':False}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['chain','audit-owners','init','new-session','import23','status','run','aggregate','aggregate-worker'])
    ap.add_argument('--root',type=Path);ap.add_argument('--index',type=Path,default=DEFAULT_INDEX);ap.add_argument('--output',type=Path)
    ap.add_argument('--chain',type=Path);ap.add_argument('--handover',type=Path);ap.add_argument('--authorization',type=Path);ap.add_argument('--diff',type=Path)
    ap.add_argument('--minutes',type=int,default=150);ap.add_argument('--release',type=Path);ap.add_argument('--episodes',default='28');args=ap.parse_args()
    root=args.root.resolve() if args.root else None
    # Only record denials in an already initialized root, never create a new
    # production root as a side effect of a refused command.
    global CLI_ROOT, CLI_COMMAND
    CLI_ROOT,CLI_COMMAND=root,args.command
    if args.command=='chain':
        value=signed_chain(ref(args.index));atomic_json(args.output,value,True);result={'parent_bundle_ref':ref(args.output),'inherited':value['inherited_total']}
    elif args.command=='audit-owners':
        chain=load(verify_ref(ref(args.chain)));c1=load(verify_ref(chain['C1_bundle_ref']));idx=load(verify_ref(chain['parent_index_ref']));roots=[c1['parent_run'],idx['root']]
        value={'schema':'r12_historical_owner_handover_v1','created_at':now(),'roots':roots,'owners':history_owners(roots),
               'historical_C22_exit_reap':'unverified','scope':'registered owner identities only; no signals; old end gaps preserved'}
        atomic_json(args.output,value,True);result={'handover_ref':ref(args.output),'registered_owners':len(value['owners'])}
    elif args.command=='init':result=initialize(root,ref(args.chain),ref(args.handover),ref(args.authorization),ref(args.diff),args.minutes)
    elif args.command=='new-session':result=renew(root,ref(args.authorization),ref(args.diff),args.minutes,ref(args.release) if args.release else None)
    elif args.command=='import23':result=do_import(root)
    elif args.command=='status':result=status(root)
    elif args.command=='run':
        episodes=[23,24,25,26,27,29] if args.episodes=='remaining6' else [int(e) for e in args.episodes.split(',')]
        require(episodes and len(set(episodes))==len(episodes),'invalid episode selection');result=run(root,episodes)
    elif args.command=='aggregate':result=aggregate(root)
    else:result=aggregate_worker(root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        root=globals().get('CLI_ROOT')
        if root and (root/'run_manifest.json').exists():
            try:
                man=load(root/'run_manifest.json')
                atomic_json(root/'admission_denied'/f'{uuid.uuid4().hex}.json',
                    {'schema':'r12_admission_denied_v1','created_at':now(),'command':globals().get('CLI_COMMAND'),
                     'session_ref':man.get('session_ref'),'code':man.get('code'),'config_ref':man.get('config'),
                     'error_type':type(exc).__name__,'error':repr(exc),'model_start_authorized':False},True)
            except BaseException:pass
        raise
