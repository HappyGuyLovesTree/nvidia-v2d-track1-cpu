"""Real C1+C2/input/import smoke with an isolated registry and no production release."""
from pathlib import Path
from unittest.mock import patch
import argparse, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from r11_state import *
import r12
import r12_session as ses

def main(out,authorization,diff):
    out.mkdir(parents=True,exist_ok=False);root=out/'fixture_run';control=out/'fixture_control'
    chain=r12.signed_chain(ref(r12.DEFAULT_INDEX));cp=out/'parent_chain.json';atomic_json(cp,chain,True)
    c1=load(verify_ref(chain['C1_bundle_ref']));idx=load(verify_ref(chain['parent_index_ref']));roots=[c1['parent_run'],idx['root']]
    hp=out/'historical_handover.json';atomic_json(hp,{'schema':'r12_historical_owner_handover_v1','created_at':now(),'roots':roots,'owners':ses.history_owners(roots)},True)
    with patch.object(r12,'global_directory',lambda cfg:control),patch.object(ses,'global_directory',lambda cfg:control):
        r12.initialize(root,ref(cp),ref(hp),ref(authorization),ref(diff),30)
        imported=r12.do_import(root);before=load(root/'budget/state.json')
        r12.renew(root,ref(authorization),ref(diff),30)
        after=load(root/'budget/state.json');status=r12.status(root)
        require(before['counts']==after['counts']==chain['inherited_total']['counts'],'counts changed without model calls')
        require(all(after['cumulative_write_bytes'].get(k,0)>=v for k,v in before['cumulative_write_bytes'].items()),'renewal refunded bytes')
        require(after['written_bytes']['C']-chain['inherited_total']['written_bytes']['C']==3646035,'real final-layer copy size mismatch')
        require(status['completed_legacy']==list(range(23)) and status['completed_new']==[],'smoke completion scope')
        require(status['projection']['projected_total']['rig']==20234,'remaining projection')
        require(not (root/'budget/calls.jsonl').exists(),'import fabricated calls')
        try:r12.initialize(out/'second_root',ref(cp),ref(hp),ref(authorization),ref(diff),30)
        except ValueError as exc:crossroot=str(exc)
        else:raise AssertionError('new root silently reset budget')
        try:r12.run(root,[28])
        except ValueError as exc:production_denied=str(exc)
        else:raise AssertionError('unreleased model entry allowed')
    atomic_json(out/'result.json',{'schema':'T012.05_real_chain_import_smoke_v1','created_at':now(),'passed':True,
         'isolated_registry_for_fixture_only':str(control),'run_root':str(root),'import':imported,
         'copied_artifact_bytes':3646035,'counts':after['counts'],
         'cross_root_denied':crossroot,'production_denied':production_denied,'renewal_preserves_counts_and_bytes':True,
         'real_model_loads':0,'real_model_forwards':0,'real_video_decodes':0},True)
    print('passed real parent chain + 23 import + renewal + cross-root/production denial; fixture registry isolated')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--authorization',type=Path,required=True);ap.add_argument('--diff',type=Path,required=True);a=ap.parse_args();main(a.output.resolve(),a.authorization,a.diff)
