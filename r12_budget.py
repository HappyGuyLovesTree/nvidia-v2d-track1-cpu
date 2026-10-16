"""Read-only C1+C2 chain validation; current journals never include ancestor calls."""
from pathlib import Path
from collections import Counter
import json
from common import load
from tracking import digest
from r11_state import require, ref, verify_ref, now
from r11_budget import FIELDS, validate_parent

IDENTITY=('model_kind','episode','frame','attempt','stage','input_identity','ordinal')
C2='483f8d61690d7df229c57dca43750b23afcfbf57'
TREE='488470c7217dedb23e72abe70e896ca4b3e078d8'
INDEX_SHA='23f3e515bd54fadc8f5617c13f8f55c92374dd47d87129296c6b4d2b50db4475'

def delta(total,base):
    out={k:v-base.get(k,0) for k,v in total.items()}
    require(all(type(v) is int and v>=0 for v in out.values()),'total below baseline')
    require(all(k in total or v==0 for k,v in base.items()),'baseline key missing')
    return {k:v for k,v in out.items() if v}

def validate_journal(state,path,baseline=None,ancestor_ids=(),allow_pending=False):
    baseline=baseline or {};starts={};ends={};counts=Counter();outcomes=Counter();episodes=Counter()
    last=dict(baseline.get('counts',{}))
    if Path(path).exists():
        with Path(path).open('rb') as stream:
            for line in stream:
                require(line.endswith(b'\n'),'torn call journal')
                row=json.loads(line);cid=row['call_id'];kind=row['model_kind']
                if row['kind']=='call_start':
                    require(cid not in starts and cid not in ancestor_ids,'duplicate/ancestor call_id')
                    require(type(row['ordinal']) is int and row['ordinal']==last.get(kind,0)+1,'non-contiguous ordinal')
                    require(kind in ('detector','hmr','rig','dnn_load','ort_session_load','rig_load'),'unknown model kind')
                    last[kind]=row['ordinal'];starts[cid]=row;counts[kind]+=1;episodes[str(row['episode'])+':'+kind]+=1
                elif row['kind']=='call_end':
                    require(cid in starts and cid not in ends,'unmatched/duplicate call_end')
                    require(all(row[k]==starts[cid][k] for k in IDENTITY),'call identity mismatch')
                    require(row['outcome'] in ('returned','failed'),'unknown outcome')
                    ends[cid]=row;outcomes[kind+':'+row['outcome']]+=1
                else:raise ValueError('unknown journal event')
    pending={cid:r for cid,r in starts.items() if cid not in ends}
    require(pending==state.get('pending',{}),'pending/journal mismatch')
    require(allow_pending or not pending,'unclosed global call, including load')
    require(not state.get('removals'),'unresolved removals')
    require(not state.get('stop_reason'),'latched resource stop')
    require(dict(counts)==delta(state['counts'],baseline.get('counts',{})),'counts/journal mismatch')
    require(dict(outcomes)==delta(state.get('outcomes',{}),baseline.get('outcomes',{})),'outcomes/journal mismatch')
    require(dict(episodes)=={k:v for k,v in state.get('episode_counts',{}).items() if v},'episode counts/journal mismatch')
    for field in ('written_bytes','cumulative_write_bytes'):
        delta(state[field],baseline.get(field,{}))
        require(all(type(v) is int and v>=0 for v in state[field].values()),'invalid byte accounting')
    require(all(state['cumulative_write_bytes'].get(k,0)>=v for k,v in state['written_bytes'].items()),'retained exceeds cumulative writes')
    return {'counts':dict(counts),'outcomes':dict(outcomes),'episode_counts':dict(episodes),
            'call_ids':set(starts),'pending':pending,'calls':len(starts)}

def chain_payload(index_ref):
    require(index_ref['sha256']==INDEX_SHA,'not approved final C2 index')
    index=load(verify_ref(index_ref));require(index['code']=={'commit':C2,'tree':TREE},'C2 producer mismatch')
    require(index['production_frozen'] and index['completed_episodes']==list(range(23)),'not final23 frozen index')
    frozen=index['frozen_files'];[verify_ref(r) for r in frozen.values()]
    verify_ref(index['config']);cfg=load(index['config']['path']);manifest=load(frozen['manifest']['path'])
    require(manifest['config']==index['config'] and manifest['code']['commit']==C2 and manifest['code']['tree']==TREE,'C2 manifest binding')
    c1ref=cfg['budget_parent'];c1=validate_parent(c1ref);b1=load(c1ref['path'])
    for r in b1['original_refs'].values():verify_ref(r)
    s1=load(b1['snapshots']['ledger']['path']);j1=validate_journal(s1,b1['snapshots']['journal']['path'])
    s2=load(frozen['ledger']['path'])
    require(s2['baseline']['source']==c1ref and all(s2['baseline'][k]==c1[k] for k in FIELDS),'C2 baseline not exact C1')
    j2=validate_journal(s2,frozen['journal']['path'],c1,j1['call_ids'])
    pipeline=load(frozen['pipeline_index']['path'])
    require(all(pipeline['cost'][k]==s2[k] for k in FIELDS),'pipeline/ledger mismatch')
    require(pipeline['cost'].get('pending',{})=={} and not pipeline['unknown_attempts'],'pipeline unknown calls')
    root=Path(index['root']);live={'ledger':ref(root/'budget/state.json'),'journal':ref(root/'budget/calls.jsonl')}
    require(all(live[k]['sha256']==frozen[k]['sha256'] for k in live),'C2 changed after frozen ledger')
    total={k:s2[k] for k in FIELDS};current={k:delta(total[k],c1[k]) for k in FIELDS}
    return {'schema':'r12_budget_parent_chain_v1','created_at':now(),'parent_index_ref':index_ref,
            'parent_config_ref':index['config'],'parent_manifest_ref':frozen['manifest'],
            'C1_bundle_ref':c1ref,'C1_original_refs':b1['original_refs'],'C1_snapshot_refs':b1['snapshots'],
            'C2_frozen_refs':frozen,'C2_live_refs':live,'C2_baseline':s2['baseline'],
            'C1_total':c1,'C2_current':current,'C2_total':total,'inherited_total':total,
            'parent_history':{'C2_episode_counts':s2['episode_counts'],'completed_episodes':list(range(23)),
                              'C1_calls':j1['calls'],'C2_calls':j2['calls']},
            'original_window_start':cfg['window_start'],'historical_C22_exit_reap':'unverified',
            'validation':{'journal_closed':True,'disjoint_call_ids':True,'ordinals_contiguous':True,
                          'unknown':False,'resource_stop':False,'removals_pending':False}}

def signed_chain(index_ref):
    value=chain_payload(index_ref);value['chain_digest']=digest(value);return value

def validate_chain(bundle_ref):
    bundle=load(verify_ref(bundle_ref));require(bundle['schema']=='r12_budget_parent_chain_v1','wrong chain schema')
    payload={k:v for k,v in bundle.items() if k!='chain_digest'}
    require(digest(payload)==bundle['chain_digest'],'chain digest mismatch')
    current=chain_payload(bundle['parent_index_ref']);current['created_at']=bundle['created_at']
    require(current==payload,'parent chain no longer matches')
    return bundle['inherited_total']

def validate_current(root,cfg):
    state=load(Path(root)/'budget/state.json');base=state['baseline']
    require(base['source']==cfg['budget_parent'],'current baseline source changed')
    bundle=load(verify_ref(cfg['budget_parent']))
    require(all(base[k]==bundle['inherited_total'][k] for k in FIELDS),'current baseline tampered')
    result=validate_journal(state,Path(root)/'budget/calls.jsonl',base)
    return state,result
