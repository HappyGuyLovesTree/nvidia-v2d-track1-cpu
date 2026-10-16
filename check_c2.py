"""C2 targeted negative cases. No model imports, loads, forwards or video decode."""
from pathlib import Path
from unittest.mock import patch
import argparse,copy,datetime,json,tempfile
from r11_state import *
from r11_budget import validate_parent
from common import load
from r11_worker import stage_a,stage_b
from r11_pack import stage_c
from r11 import run,prepare

def reject(fn):
    try:fn()
    except (BudgetStop,ValueError):return
    raise AssertionError('expected rejection')
def main(out):
    cfg=load(Path(__file__).parent/'config_r11_20261017a1.json');cfg=copy.deepcopy(cfg);future=(datetime.datetime.now(TZ)+datetime.timedelta(days=1)).isoformat();cfg['deadlines']={k:future for k in cfg['deadlines']};cases=[]
    with tempfile.TemporaryDirectory(prefix='r11_c2_checks_') as td:
        root=Path(td)
        for i,kind in enumerate(('dnn_load','ort_session_load','rig_load','detector','hmr','rig')):
            c=copy.deepcopy(cfg);c.pop('budget_parent');l=Ledger(root/str(i),c);w=Writer(l)
            try:l.invoke(kind,lambda:(_ for _ in ()).throw(BudgetStop('synthetic unclosed')),episode=16,frame=None if kind.endswith('load') else 51,attempt='t',stage='B' if kind.startswith('rig') else 'A',n=240)
            except BudgetStop:pass
            before=l.snapshot()['counts'].copy();called=[]
            reject(lambda:l.invoke('hmr',lambda:called.append(1),episode=0,frame=99,attempt='next',stage='A',n=240));assert not called and l.snapshot()['counts']==before
            for f in (stage_a,stage_b,stage_c):reject(lambda:f(l.root,l.root/'er',{},c,{},l,w,None))
            # run also rejects even when there are no selected episodes/all stages would skip.
            reject(lambda:run(l.root,c,{},[]));cases.append('global_unknown_blocks_all_entrypoints:'+kind)
        c=copy.deepcopy(cfg);c.pop('budget_parent');l=Ledger(root/'cleanup',c);w=Writer(l);p=l.root/'prepared/f.json';w.bytes(p,b'x'*100,'A');before=l.snapshot()
        real_unlink=Path.unlink
        def crash_after_unlink(path,*a,**kw):
            result=real_unlink(path,*a,**kw)
            if path==p.resolve():raise BudgetStop('synthetic after unlink before accounting')
            return result
        with patch.object(Path,'unlink',crash_after_unlink):reject(lambda:w.remove(p,'A'))
        assert not p.exists();after=l.snapshot();assert after['written_bytes']['A']==0 and after['cumulative_write_bytes']==before['cumulative_write_bytes'];w.remove(p,'A');assert l.snapshot()['written_bytes']['A']==0;cases.append('cleanup_crash_after_unlink_idempotent_no_cumulative_reduction')
        # Failure before unlink must retain both file and charge.
        w.bytes(p,b'x'*100,'A')
        with patch.object(Path,'unlink',lambda *a,**kw:(_ for _ in ()).throw(OSError('before unlink'))):
            try:w.remove(p,'A')
            except OSError:pass
        assert p.exists() and l.snapshot()['written_bytes']['A']==100;w.remove(p,'A');cases.append('cleanup_before_unlink_failure_preserves_charge')
        inherited=Ledger(root/'inherited',cfg);s=inherited.snapshot();base=validate_parent(cfg['budget_parent']);assert s['counts']==base['counts'] and s['episode_counts']=={} and s['pending']=={};assert s['written_bytes']==base['written_bytes'] and s['cumulative_write_bytes']==base['cumulative_write_bytes'];assert s['accounting']['current']['counts']=={k:0 for k in base['counts']};cases.append('C1_totals_inherited_current_episode_and_completion_empty')
        _,end=inherited.invoke('detector',lambda:123,episode=16,frame=50,attempt='synthetic',stage='A',n=240);s=inherited.snapshot();assert s['counts']['detector']==241 and s['episode_counts']['16:detector']==1 and s['accounting']['current']['counts']['detector']==1;assert read_line_ref(end)['ordinal']==241;cases.append('new_run_charges_after_baseline_no_episode_cap_contamination')
        # Exhausting inherited total must not get another forward in a new episode.
        with inherited.edit() as state:state['counts']['detector']=cfg['call_limits']['detector']
        called=[];reject(lambda:inherited.invoke('detector',lambda:called.append(1),episode=1,frame=1,attempt='t',stage='A',n=240));assert not called;cases.append('global_total_limit_not_reset_by_new_episode')
        bundle=load(cfg['budget_parent']['path']);bad=copy.deepcopy(bundle);bad['baseline']['counts']['hmr']=0;bp=root/'tampered.json';atomic_json(bp,bad);reject(lambda:validate_parent(ref(bp)));cases.append('parent_baseline_tamper_rejected')
        ledger_path=Path(bundle['snapshots']['ledger']['path']);original=load(ledger_path);badstate=copy.deepcopy(original);badstate['pending']={'bad':{'frame':None}};badledger=root/'badledger.json';atomic_json(badledger,badstate);bad=copy.deepcopy(bundle);bad['snapshots']['ledger']=ref(badledger);atomic_json(bp,bad);reject(lambda:validate_parent(ref(bp)));cases.append('parent_unknown_rejected')
        reject(lambda:prepare(root/'unauthorized_new_root',Path(__file__).parent/'config_r11_20261017a1.json'));assert not (root/'unauthorized_new_root/budget').exists();cases.append('prepare_rejects_new_budget_reset_root_before_mutation')
        import subprocess,sys
        badroot=root/'cli_unauthorized';proc=subprocess.run([sys.executable,'-B',str(Path(__file__).parent/'r11.py'),'--stage','prepare','--output',str(badroot)],capture_output=True,text=True);assert proc.returncode!=0 and not badroot.exists();cases.append('CLI_wrong_root_rejected_before_owner_files')
        result={'schema':'C2_targeted_checks_v1','status':'passed','as_of':now(),'cases':cases,'groups':len(cases),'parent':cfg['budget_parent'],'baseline':base,'real_model_loads':0,'real_model_forwards':0,'video_decodes':0,'source_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('r11*.py')}};atomic_json(out,result,True);print(json.dumps({'status':'passed','groups':len(cases),'output':str(out)}))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();main(a.output)
