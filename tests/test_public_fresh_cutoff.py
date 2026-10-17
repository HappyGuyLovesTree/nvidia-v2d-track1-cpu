"""I12-PUB02: synthetic official admission only; all inference work is intercepted.

Fixture preparation uses the existing fake probe, never an actual video decoder.
No completed prediction, official input readiness or full fresh replay is claimed.
"""
from pathlib import Path
from unittest.mock import patch
import datetime,signal,sys,time,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import r11_inputs
import r11_state as st
import r11_worker
from v2d_public import fresh,fresh_worker
from v2d_public.common import load,save_new,ContractError
import test_public_fresh as fixture_module

class FreshCutoff(unittest.TestCase):
    def setUp(self):
        helper=fixture_module.FreshIntegration('test_fresh_init_bridge_empty_parent_and_no_calls');helper.setUp()
        self.root=helper.fixture_run()[0]
        man=fresh.verify_fresh_root(self.root);cfg=load(man['config']['path'])
        # Same explicit synthetic scope conversion as the independent T91 counterexample.
        # Real official input/asset preparation is outside this admission fixture.
        cfg.update(scope='official',assets={})
        man.update(scope='official',config=save_new(self.root/'official_config_SYNTHETIC.json',cfg))
        st.atomic_json(self.root/'run_manifest.json',man)
        self.session=load(st.verify_ref(load(self.root/'session_latest.json')))
        self.session['scope']='official';self.serial=0
        self.deadlines(new=60,candidate=120,hard=180)

    def deadlines(self,*,new,candidate,hard):
        mono=time.monotonic();wall=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
        for key,seconds in [('new_episode_stop',new),('candidate_stop',candidate),('hard_stop',hard)]:
            self.session['monotonic_deadlines'][key]=mono+seconds
            self.session[key]=(wall+datetime.timedelta(seconds=seconds)).isoformat()
        self.serial+=1
        st.atomic_json(self.root/'session_latest.json',save_new(self.root/'sessions'/f'cutoff_SYNTHETIC_{self.serial}.json',self.session))

    def controller(self,*,completed_a=False,before_preflight=None):
        requests=[]
        def attempt(**kw):
            if before_preflight:before_preflight(kw)
            kw['preflight']()
            requests.append(kw['stage'])
            return {'supervisor_outcome':'passed'}
        def observe(root,episode,stage):
            return {'status':'completed' if completed_a and stage=='A' else 'absent'}
        with patch.object(fresh,'observe_artifact',side_effect=observe),patch.object(fresh,'validate_stage',return_value={'SYNTHETIC_COMPLETED_A':True}),patch.object(fresh,'execute_attempt',side_effect=attempt):
            result=fresh.execute(self.root,[0],allow_models=True)
        return result,requests

    def assert_no_leases_or_calls(self):
        self.assertEqual(list((self.root/'leases').glob('*.json')),[])
        cfg,_=fresh.runtime_config(self.root)
        state,_=fresh.validate_budget(self.root,cfg)
        self.assertEqual(state['counts'],{})
        self.assertFalse((self.root/'budget/calls.jsonl').exists())

    def test_expired_new_episode_refuses_before_lease_or_request(self):
        self.deadlines(new=-1,candidate=120,hard=180)
        with patch.object(fresh,'execute_attempt') as attempt:
            with self.assertRaises(fresh.SupervisionFailure) as caught:
                fresh.execute(self.root,[0],allow_models=True)
        self.assertEqual(caught.exception.record['errors']['primary_error']['message'],'fresh_new_episode_stop')
        self.assertIsInstance(caught.exception.__cause__,st.BudgetStop)
        attempt.assert_not_called();self.assert_no_leases_or_calls()

    def test_open_window_requests_a_b_c(self):
        result,requests=self.controller()
        self.assertEqual(requests,['A','B','C'])
        self.assertEqual(result['mode'],'actual_fresh_inference')
        self.assertFalse((self.root/'budget/calls.jsonl').exists())

    def test_completed_a_can_finish_b_c_after_new_episode_stop(self):
        self.deadlines(new=-1,candidate=120,hard=180)
        _,requests=self.controller(completed_a=True)
        self.assertEqual(requests,['B','C'])

    def test_candidate_stop_refuses_b_after_completed_a(self):
        self.deadlines(new=-2,candidate=-1,hard=180)
        with self.assertRaises(fresh.SupervisionFailure) as caught:
            self.controller(completed_a=True)
        self.assertEqual(caught.exception.record['errors']['primary_error']['message'],'fresh_monotonic_candidate_stop')
        self.assert_no_leases_or_calls()

    def test_hard_stop_refuses_controller(self):
        self.deadlines(new=-3,candidate=-2,hard=-1)
        with patch.object(fresh,'execute_attempt') as attempt:
            with self.assertRaisesRegex(ContractError,'fresh session expired'):
                fresh.execute(self.root,[0],allow_models=True)
        attempt.assert_not_called()
        self.assertEqual(list((self.root/'leases').glob('*.json')),[])

    def test_new_episode_crossed_during_queue_refuses_preflight(self):
        # Keep real monotonic progression except at the deterministic crossing.
        # The lease has been issued, but the supervisor preflight must not spawn.
        deadline=self.session['monotonic_deadlines']['new_episode_stop']
        def queued(kw):
            self.assertEqual(kw['stage'],'A')
            with patch.object(fresh.time,'monotonic',return_value=deadline):
                kw['preflight']()
        with self.assertRaises(fresh.SupervisionFailure) as caught:
            self.controller(before_preflight=queued)
        self.assertEqual(caught.exception.record['errors']['primary_error']['message'],'fresh_new_episode_stop')
        self.assertEqual(len(list((self.root/'leases').glob('*.json'))),1)

    def test_deadline_equality_and_wall_rollback_use_monotonic(self):
        cfg,_=fresh.runtime_config(self.root)
        cfg['fresh_session']['monotonic_deadlines'].update(new_episode_stop=100,candidate_stop=200,hard_stop=300)
        cfg['fresh_session']['new_episode_stop']='2099-01-01T00:00:00+08:00'
        ledger=fresh.FreshLedger(self.root,cfg)
        with patch.object(fresh.time,'monotonic',return_value=99.999):ledger.check_time(model=True,new_episode=True)
        for instant,kwargs,reason in [(100,{'new_episode':True},'fresh_new_episode_stop'),(200,{'model':True},'fresh_monotonic_candidate_stop'),(300,{},'fresh_monotonic_hard_stop')]:
            with self.subTest(instant=instant),patch.object(fresh.time,'monotonic',return_value=instant):
                with self.assertRaisesRegex(st.BudgetStop,reason):ledger.check_time(**kwargs)

    def worker(self,stage):
        argv=['--root',str(self.root),'--episode','0','--stage',stage,'--lease',str(self.root/'SYNTHETIC_LEASE'),'--lease-hash','SYNTHETIC']
        # Parent lease and input checks are separately covered by existing tests.
        # Here isolate the real worker dispatch/time gate; numerical work is a spy.
        with patch.object(fresh_worker,'validate_lease'),patch.object(r11_inputs,'verify_episode'),patch.object(signal,'signal'),patch.object(signal,'setitimer'),patch.object(r11_worker,'stage_a',return_value={'SYNTHETIC':True}) as a,patch.object(r11_worker,'stage_b',return_value={'SYNTHETIC':True}) as b,patch.object(fresh_worker,'dense_stage',return_value={'SYNTHETIC':True}) as c:
            failure=None
            try:result=fresh_worker.main(argv)
            except fresh.SupervisionFailure as exc:
                self.assertIsInstance(exc.__cause__,st.BudgetStop)
                self.assertEqual(exc.record['supervisor_outcome'],'failed')
                failure=exc.__cause__;result=None
        return result,failure,[a.call_count,b.call_count,c.call_count]

    def test_worker_a_rechecks_before_first_work(self):
        self.deadlines(new=-1,candidate=120,hard=180)
        _,failure,calls=self.worker('A')
        self.assertEqual(str(failure),'fresh_new_episode_stop');self.assertEqual(calls,[0,0,0])
        receipt=load(st.verify_ref(load(self.root/'episode_runs/episode_000000/A_latest.json')))
        self.assertEqual(receipt['status'],'failed')
        self.assertEqual(receipt['primary_error']['message'],'fresh_new_episode_stop')
        self.assertFalse((self.root/'budget/calls.jsonl').exists())

    def test_worker_a_in_open_window_dispatches(self):
        result,failure,calls=self.worker('A')
        self.assertEqual(result,0);self.assertIsNone(failure);self.assertEqual(calls,[1,0,0])

    def test_worker_b_c_finish_after_new_episode_stop(self):
        self.deadlines(new=-1,candidate=120,hard=180)
        for stage,expected in [('B',[0,1,0]),('C',[0,0,1])]:
            with self.subTest(stage=stage):
                result,failure,calls=self.worker(stage)
                self.assertEqual(result,0);self.assertIsNone(failure);self.assertEqual(calls,expected)

if __name__=='__main__':unittest.main(verbosity=2)
