"""SES/BUD/IMP counterexamples. No real models, video readers or production workers."""
from pathlib import Path
from unittest.mock import patch
import argparse, copy, datetime, errno, fcntl, io, json, os, sys, tempfile, time, unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import load
from tracking import digest
from r11_state import *
import r12_budget as budget
import r12_session as ses
import r12_import as imp
import r12_supervision as sup
import r12

OUT=None;CASES=[]

def guard(event,args):
    if event=='import' and args[0].split('.')[0] in ('torch','cv2','av','onnxruntime'):raise AssertionError('model/decoder import forbidden')
    if event in ('socket.connect','socket.bind','os.kill','os.killpg','subprocess.Popen'):raise AssertionError('process/network/signal forbidden')
sys.addaudithook(guard)

class Cases(unittest.TestCase):
    def setUp(self):
        self.root=OUT/self.id().split('.')[-1];self.root.mkdir();self.journal=self.root/'calls.jsonl'
        self.base={'counts':{'detector':2},'outcomes':{'detector:returned':2},'written_bytes':{'C':10},'cumulative_write_bytes':{'C':20}}
        self.start={'kind':'call_start','call_id':'new-id','model_kind':'detector','episode':28,'frame':90,'attempt':'fixture','stage':'A','input_identity':'fixture-input','ordinal':3}
        self.end={**self.start,'kind':'call_end','outcome':'returned'}
        self.state={'counts':{'detector':3},'outcomes':{'detector:returned':3},'episode_counts':{'28:detector':1},'pending':{},'removals':{},'written_bytes':{'C':10},'cumulative_write_bytes':{'C':20}}
        self.rows=[self.start,self.end];self.write()
    def write(self):self.journal.write_bytes(b''.join(json_bytes(r)+b'\n' for r in self.rows))
    def check(self,**kw):return budget.validate_journal(self.state,self.journal,self.base,**kw)
    def test_closed_once_and_next_ordinal(self):
        self.assertEqual(self.check()['calls'],1);self.assertEqual(self.start['ordinal'],3)
    def test_duplicate_inheritance_rejected(self):
        self.state['counts']['detector']+=2;self.assertRaises(ValueError,self.check)
    def test_ancestor_call_id_overlap(self):self.assertRaises(ValueError,self.check,ancestor_ids={'new-id'})
    def test_duplicate_start(self):self.rows.insert(1,self.start);self.write();self.assertRaises(ValueError,self.check)
    def test_duplicate_end(self):self.rows.append(self.end);self.write();self.assertRaises(ValueError,self.check)
    def test_torn_journal(self):self.journal.write_bytes(self.journal.read_bytes().rstrip());self.assertRaises(ValueError,self.check)
    def test_ordinal_gap(self):
        self.rows[0]['ordinal']=4;self.rows[1]['ordinal']=4;self.write();self.assertRaises(ValueError,self.check)
    def test_end_identity_changed(self):self.rows[1]['stage']='B';self.write();self.assertRaises(ValueError,self.check)
    def test_failure_stays_charged(self):
        self.rows[1]['outcome']='failed';self.state['outcomes']={'detector:returned':2,'detector:failed':1};self.write();self.assertEqual(self.check()['calls'],1)
    def test_unknown_load_frame_null_blocks(self):
        self.start.update(model_kind='rig_load',frame=None,ordinal=1);self.rows=[self.start];self.write()
        self.state.update(counts={'detector':2,'rig_load':1},outcomes={'detector:returned':2},episode_counts={'28:rig_load':1},pending={'new-id':self.start})
        self.assertRaises(ValueError,self.check);self.assertEqual(self.check(allow_pending=True)['calls'],1)
    def test_pending_missing_from_state(self):self.rows=[self.start];self.write();self.assertRaises(ValueError,self.check)
    def test_removal_blocks(self):self.state['removals']={'x':{}};self.assertRaises(ValueError,self.check)
    def test_resource_stop_blocks(self):self.state['stop_reason']='RSS';self.assertRaises(ValueError,self.check)
    def test_byte_refund_below_parent(self):self.state['written_bytes']['C']=9;self.assertRaises(ValueError,self.check)
    def test_cumulative_never_refunded(self):self.state['cumulative_write_bytes']['C']=19;self.assertRaises(ValueError,self.check)
    def test_cross_root_registry_rejected(self):
        atomic_json(self.root/'registry.json',{'active_root':str(self.root),'parent_bundle_ref':{'hash':'fixture'}})
        ses.check_registry(self.root,self.root,{'hash':'fixture'})
        self.assertRaises(ValueError,ses.check_registry,self.root,self.root/'new',{'hash':'fixture'})
    def test_kernel_lock_contention(self):
        cfg={'root':str(self.root)}
        with r12.registry_lock(cfg):self.assertRaises(BlockingIOError,lambda: r12.registry_lock(cfg).__enter__())
    def session(self):
        p=self.root/'reference.json';atomic_json(p,{'fixture':True});rr=ref(p);t=datetime.datetime.now(TZ)
        return {'schema':ses.SESSION,'mode':'production_resume','session_id':'fixture','active_root':str(self.root),
                'authorization_refs':[rr],'parent_bundle_ref':rr,'parent_index_ref':rr,'historical_handover_ref':rr,'old_new_diff_ref':rr,
                'original_window_start':'2026-10-17T01:06:03+08:00','controller_code':{'commit':'f'*40},'engine_code':{'commit':'f'*40},
                'start_at':(t-datetime.timedelta(seconds=1)).isoformat(),'new_episode_stop':(t+datetime.timedelta(seconds=100)).isoformat(),
                'candidate_stop':(t+datetime.timedelta(seconds=150)).isoformat(),'freeze_at':(t+datetime.timedelta(seconds=180)).isoformat(),
                'hard_stop':(t+datetime.timedelta(seconds=210)).isoformat(),'cleanup':ses.CLEANUP}
    def test_valid_session_and_expiry(self):
        s=self.session();ses.validate_session(s,self.root);s['hard_stop']=(datetime.datetime.now(TZ)-datetime.timedelta(seconds=1)).isoformat();self.assertRaises(ValueError,ses.validate_session,s,self.root)
    def test_global_baseline_not_session_start(self):
        s=self.session();s['original_window_start']=s['start_at'];self.assertRaises(ValueError,ses.validate_session,s,self.root)
    def test_fresh_never_in_resume(self):s=self.session();s['mode']='fresh_reproduction';self.assertRaises(ValueError,ses.validate_session,s,self.root)
    def test_cleanup_and_freeze_headroom(self):s=self.session();s['freeze_at']=s['candidate_stop'];self.assertRaises(ValueError,ses.validate_session,s,self.root)
    def test_owner_live_or_unknown_not_recovered(self):
        import psutil
        self.assertRaises(ValueError,ses.owner_observation,{'pid':os.getpid(),'pid_started':psutil.Process().create_time()})
        self.assertRaises(ValueError,ses.owner_observation,{'pid':os.getpid(),'pid_started':None})
    def test_missing_controller_terminal_blocks_global(self):
        atomic_json(self.root/'a.started.json',{'schema':'r12_controller_finalization_v1','supervisor_outcome':'unknown'})
        self.assertRaises(ValueError,ses.assert_closed,self.root)
    def test_controller_failed_blocks_global(self):
        atomic_json(self.root/'a.started.json',{'schema':'r12_controller_finalization_v1','supervisor_outcome':'unknown'})
        atomic_json(self.root/'a.json',{'schema':'r12_controller_finalization_v1','supervisor_outcome':'failed'})
        self.assertRaises(ValueError,ses.assert_closed,self.root)
    def test_copy_charged_and_hash_and_no_hardlink(self):
        cfg={'deadlines':{'hard_stop':(datetime.datetime.now(TZ)+datetime.timedelta(minutes=1)).isoformat()},'storage_limits':{'C':10000},'global_new_file_limit':2**32}
        ledger=Ledger(self.root/'run',cfg);writer=Writer(ledger);src=self.root/'original';src.write_bytes(b'fixture-original')
        out=imp.copy_checked(ref(src),self.root/'run/copy',writer)
        self.assertNotEqual(src.stat().st_ino,Path(out['path']).stat().st_ino)
        self.assertEqual(ledger.snapshot()['cumulative_write_bytes']['C'],len(b'fixture-original'))
        imp.copy_checked(ref(src),self.root/'run/copy',writer);self.assertEqual(ledger.snapshot()['written_bytes']['C'],len(b'fixture-original'))
        Path(out['path']).write_bytes(b'changed');self.assertRaises(ValueError,imp.copy_checked,ref(src),out['path'],writer)
    def test_hardlink_refused(self):
        src=self.root/'original';src.write_bytes(b'x');os.link(src,self.root/'copy')
        self.assertRaises(ValueError,imp.copy_checked,ref(src),self.root/'copy',None)
    def test_real_frozen_chain_and_exact_remaining(self):
        chain=budget.signed_chain(ref(r12.DEFAULT_INDEX));self.assertEqual(chain['C2_total'],chain['inherited_total'])
        self.assertEqual(chain['inherited_total']['counts']['detector'],8319);self.assertEqual(chain['inherited_total']['counts']['rig'],16638)
        self.assertEqual(chain['parent_history']['C1_calls'],963);self.assertEqual(chain['parent_history']['C2_calls'],32385)
        path=self.root/'chain.json';atomic_json(path,chain);budget.validate_chain(ref(path))
        chain['inherited_total']['counts']['rig']+=1;atomic_json(path,chain);self.assertRaises(ValueError,budget.validate_chain,ref(path))
    def test_projection_full_remaining_and_cap(self):
        idx=load(r12.DEFAULT_INDEX);cfg=load(idx['config']['path']);parent=load(idx['frozen_files']['ledger']['path'])
        base={k:parent[k] for k in budget.FIELDS};bundle=self.root/'parent.json';atomic_json(bundle,{'inherited_total':base})
        cfg={**cfg,'budget_parent':ref(bundle)}
        state={**copy.deepcopy(base),'baseline':{'source':ref(bundle),**copy.deepcopy(base)},'pending':{},'removals':{},'episode_counts':{}}
        atomic_json(self.root/'budget/state.json',state)
        oldroot=Path(idx['root'])
        for e in range(23,30):atomic_bytes(self.root/'inputs'/f'episode_{e:06d}.json',(oldroot/'inputs'/f'episode_{e:06d}.json').read_bytes(),True)
        projected=r12.projection(self.root,cfg,set())
        self.assertEqual(projected['needed']['detector'],1798);self.assertEqual(projected['needed']['rig'],3596)
        self.assertEqual(projected['projected_total']['rig'],20234);self.assertEqual(projected['projected_total']['dnn_load'],31)
        cfg['call_limits']={**cfg['call_limits'],'rig':20233};self.assertRaises(ValueError,r12.projection,self.root,cfg,set())
    def test_explicit_runtime_config_time_propagation(self):
        s=self.session();cfg={'root':str(self.root),'window_start':s['original_window_start'],'global_new_file_limit':2**32,
                             'call_limits':{'detector':10240},'storage_limits':{'C':1000},'stage_limits':{'A':{'wall_s':2400}}}
        s.update(call_caps=cfg['call_limits'],storage_caps=cfg['storage_limits'],stage_limits=cfg['stage_limits'],
                 global_occupancy_policy={'root':str(self.root/'DD'),'window_start':cfg['window_start'],'cap':2**32})
        cp=self.root/'cfg.json';sp=self.root/'session.json';atomic_json(cp,cfg);atomic_json(sp,s)
        man={'schema':'r12_run_v1','config':ref(cp),'session_ref':ref(sp),'code':s['controller_code']}
        effective=ses.runtime_config(man,self.root)
        self.assertEqual(effective['deadlines']['hard_stop'],s['hard_stop']);self.assertEqual(effective['budget_parent'],s['parent_bundle_ref'])
        self.assertEqual(effective['window_start'],'2026-10-17T01:06:03+08:00')
        self.assertEqual(load(cp),cfg)
    def test_required_or_source_tamper_rejected(self):
        idx=load(r12.DEFAULT_INDEX);cfg=load(idx['config']['path']);roster=load(cfg['assets']['roster']['path'])
        ep=load(Path(idx['root'])/'inputs/episode_000016.json');ep['sources']=[[50]];path=self.root/'input.json';atomic_json(path,ep)
        self.assertRaises(ValueError,imp.verify_frozen_input,ref(path),roster['episodes'][16],cfg['assets']['roster'])
    def real16(self):
        idx=load(r12.DEFAULT_INDEX);cfg=load(idx['config']['path']);pipe=load(idx['frozen_files']['pipeline_index']['path'])
        return imp.legacy_entry(pipe['episodes'][16],cfg,ref(r12.DEFAULT_INDEX),idx['frozen_files']['manifest'])
    def test_old16_import_without_raw_tracker_models(self):
        e=self.real16();self.assertEqual(e['prediction_producer_commit'],budget.C2);self.assertEqual(len(e['required']),240)
        self.assertNotIn('raw',e['source_refs']);self.assertNotIn('tracker',e)
    def test_mask_changed_rejected(self):
        e=self.real16();rr=copy.deepcopy(e['source_refs']);z=read_npz(rr['mask']['path']);z['mask'][0]=True;p=self.root/'mask.npz';p.write_bytes(npz_bytes(**z));rr['mask']=ref(p)
        self.assertRaises(ValueError,imp.verify_layers,rr,e['required'])
    def test_wrong_producer_rejected_before_import(self):
        idx=load(r12.DEFAULT_INDEX);pipe=load(idx['frozen_files']['pipeline_index']['path']);row=copy.deepcopy(pipe['episodes'][16]);c=load(row['stages']['C']['receipt']['path']);c['producer']='f'*40
        p=self.root/'C.json';atomic_json(p,c);row['stages']['C']['receipt']=ref(p)
        self.assertRaises(ValueError,imp.legacy_entry,row,load(idx['config']['path']),ref(r12.DEFAULT_INDEX),idx['frozen_files']['manifest'])
    def test_23_plus_7_union_and_overlap(self):
        roster=load(load(load(r12.DEFAULT_INDEX)['config']['path'])['assets']['roster']['path'])['episodes']
        old={r['episode']:r['required_original_frames'] for r in roster[:23]};new={r['episode']:r['required_original_frames'] for r in roster[23:]}
        self.assertFalse(set(old)&set(new));self.assertEqual(set(old)|set(new),set(range(30)))
        self.assertEqual(sum(map(len,old.values())),8079);self.assertEqual(sum(map(len,new.values())),1798)
        self.assertEqual(sum(50*len(v)+8231 for v in {**old,**new}.values()),740780)
        self.assertRaises(ValueError,imp.export_layers,self.root,[{'episode':16},{'episode':16}],None,{'commit':'f'*40})
    def test_terminal_published_then_save_failure_guarded(self):
        start=self.root/'attempt.started.json';atomic_json(start,{'schema':'r12_supervision_attempt_v1','attempt_id':'fixture','supervisor_outcome':'unknown'})
        def save(p,v,exclusive=False):sup.atomic_json(p,v,exclusive);raise OSError(errno.EIO,'after directory fsync')
        value={'schema':'r12_supervision_attempt_v1','attempt_id':'fixture','supervisor_outcome':'passed'}
        self.assertRaises(OSError,sup.publish_terminal,self.root/'attempt.json',value,save)
        self.assertEqual(load(self.root/'attempt.json')['supervisor_outcome'],'passed')
        self.assertEqual(sup.read_attempt(start)['supervisor_outcome'],'unknown');self.assertRaises(ValueError,sup.assert_prior_attempts_closed,self.root)
    def test_controller_terminal_directory_fsync_failure_guarded(self):
        class Owner:
            def __enter__(self):return self
            def __exit__(self,*a):pass
        def save(path,value,exclusive=False):
            if value.get('phase')=='terminal':
                real=sup.os.fsync
                def fail(fd):
                    import stat
                    if stat.S_ISDIR(os.fstat(fd).st_mode):raise OSError(errno.EIO,'directory fsync after publish')
                    return real(fd)
                with patch.object(sup.os,'fsync',fail):return sup.atomic_json(path,value,exclusive)
            return sup.atomic_json(path,value,exclusive)
        with self.assertRaises(sup.SupervisionFailure):
            with sup.control_scope(Owner(),self.root/'control',save=save,fallback=io.StringIO()):pass
        self.assertEqual(sup.read_attempt(self.root/'control.started.json')['supervisor_outcome'],'unknown')
        self.assertRaises(ValueError,ses.assert_closed,self.root)

class Result(unittest.TextTestResult):
    def startTest(self,t):self.start=time.monotonic();super().startTest(t)
    def stopTest(self,t):
        failed=any(case is t for case,_ in self.failures+self.errors);CASES.append({'name':t.id().split('.')[-1],'status':'failed' if failed else 'passed','elapsed_s':time.monotonic()-self.start});super().stopTest(t)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
    r=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    atomic_json(OUT/'results.json',{'schema':'T012.05_session_checks_v1','created_at':now(),'cases':CASES,'passed':r.wasSuccessful(),'real_model_loads':0,'real_model_forwards':0,'real_video_decodes':0,'real_processes':0,'real_signals':0,
                                 'failures':[{'case':case.id(),'traceback':tb} for case,tb in r.errors+r.failures]},True)
    sys.exit(0 if r.wasSuccessful() else 1)
