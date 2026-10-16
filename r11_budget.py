"""C2 immutable C1 budget parent validation. Never writes the parent run."""
from pathlib import Path
import argparse, collections, json
from common import load,sha
from tracking import digest
from r11_state import require,ref,verify_ref,atomic_json,atomic_bytes,now
FIELDS=('counts','outcomes','written_bytes','cumulative_write_bytes')

def validate_calls(state,path):
    starts={};ends={};counts=collections.Counter();outcomes=collections.Counter()
    with Path(path).open('rb') as f:
        for b in f:
            require(b.endswith(b'\n'),'parent call journal torn');r=json.loads(b);cid=r['call_id']
            if r['kind']=='call_start':
                require(cid not in starts,'duplicate parent call start');starts[cid]=r;counts[r['model_kind']]+=1
            elif r['kind']=='call_end':
                require(cid in starts and cid not in ends,'unmatched/duplicate parent end');s=starts[cid]
                require(all(r[k]==s[k] for k in ('model_kind','episode','frame','attempt','stage','input_identity','ordinal')),'parent call identity mismatch')
                require(r['outcome'] in ('returned','failed'),'unknown parent end outcome');ends[cid]=r;outcomes[r['model_kind']+':'+r['outcome']]+=1
            else:raise ValueError('unknown parent journal event')
    require(not state['pending'] and starts.keys()==ends.keys(),'parent has unknown/unclosed calls')
    require(dict(counts)==state['counts'] and dict(outcomes)==state['outcomes'],'parent counters differ from journal')
    require(not state.get('stop_reason'),'parent stopped on resource limit')
    for field in ('written_bytes','cumulative_write_bytes'):
        require(all(type(v) is int and v>=0 for v in state[field].values()),'invalid parent byte counters')
    return {k:state[k] for k in FIELDS}

def validate_parent(bundle_ref):
    bundle=load(verify_ref(bundle_ref));require(bundle['schema']=='r11_C1_budget_parent_v1','wrong parent bundle schema')
    for key in ('index','pipeline_snapshot','manifest','ledger','journal','cost'):verify_ref(bundle['snapshots'][key])
    s=load(bundle['snapshots']['ledger']['path']);baseline=validate_calls(s,bundle['snapshots']['journal']['path'])
    pipeline=load(bundle['snapshots']['pipeline_snapshot']['path']);require(all(pipeline['cost'][k]==s[k] for k in FIELDS) and not pipeline['cost']['pending'],'stable snapshot/ledger mismatch')
    cost=load(bundle['snapshots']['cost']['path']);require(cost['counts']==s['counts'] and cost['retained_bytes_by_writer_category']==s['written_bytes'] and cost['cumulative_write_bytes_by_category']==s['cumulative_write_bytes'],'parent cost boundary mismatch')
    require(baseline==bundle['baseline'],'parent baseline tampered')
    require(bundle['parent_commit']=='689f043d8efe65107b18471474a6bc5ba250aefa','not the approved C1 parent')
    return baseline

def snapshot_parent(index_ref,destination):
    destination=Path(destination).resolve();require(not destination.exists(),'immutable parent snapshot exists');index=load(verify_ref(index_ref))
    sources={'index':index_ref,'pipeline_snapshot':index['pipeline_snapshot'],'manifest':index['run_manifest'],'cost':index['artifacts']['cost']}
    root=Path(index['output_root']).resolve();sources['ledger']=ref(root/'budget/state.json');sources['journal']=ref(root/'budget/calls.jsonl')
    require(not destination.is_relative_to(root),'never write C1 root')
    for r in sources.values():verify_ref(r)
    s=load(sources['ledger']['path']);baseline=validate_calls(s,sources['journal']['path'])
    require(all(load(sources['pipeline_snapshot']['path'])['cost'][k]==s[k] for k in FIELDS),'stable parent snapshot differs')
    destination.mkdir(parents=True);snapshots={}
    for key,r in sources.items():
        target=destination/(key+'_20261017'+('.jsonl' if key=='journal' else '.json'));atomic_bytes(target,Path(r['path']).read_bytes(),True);snapshots[key]=ref(target);require(snapshots[key]['sha256']==r['sha256'],'parent copy differs')
    for r in sources.values():verify_ref(r)
    bundle={'schema':'r11_C1_budget_parent_v1','created_at':now(),'parent_run':str(root),'parent_commit':index['code']['commit'],'original_refs':sources,'snapshots':snapshots,'baseline':baseline,'current_episode_counts':'empty_in_C2','current_completion_sets':'empty_in_C2','scope':'C1 closed model/load counts and writer-retained/cumulative bytes, not a copy of model outputs'}
    p=destination/'budget_parent_20261017.json';atomic_json(p,bundle,True);validate_parent(ref(p));return ref(p)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--index',required=True);ap.add_argument('--sha256',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();print(json.dumps(snapshot_parent({'path':a.index,'sha256':a.sha256},a.output),indent=2))
