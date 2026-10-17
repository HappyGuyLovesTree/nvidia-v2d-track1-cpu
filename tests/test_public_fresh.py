"""Fresh integration: fixture inputs/ledger/lease/supervision, no real models/video."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast,contextlib,io,json,os,sys,tempfile,time,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import r11_state as st
import r11_pack
from v2d_public import fresh
from v2d_public.fresh_worker import dense_stage,validate_lease
from v2d_public.common import Paths,load,ref,save_new,sha,digest,ContractError
from v2d_public.inputs import prepare_one
from test_public_tools import fixture_row,fixture_probe

class FreshIntegration(unittest.TestCase):
    def setUp(self):
        base=os.environ.get('V2D_TEST_OUTPUT');Path(base).mkdir(parents=True,exist_ok=True) if base else None
        self.root=Path(tempfile.mkdtemp(prefix=self._testMethodName+'_',dir=base));self.paths=Paths(self.root/'workspace')
    def fixture_run(self):
        video=self.root/'fixture.bin';video.write_bytes(b'fixture bytes, not a video');row=fixture_row()
        executable=self.root/'probe_fixture';executable.write_text('#!'+sys.executable+'\nprint('+repr(json.dumps(fixture_probe()))+')\n');executable.chmod(0o700)
        prepared=prepare_one(row,video,self.paths,ffprobe=executable,scope='fixture')
        index=self.root/'preparation.json';save_new(index,{'schema':'r12_public_input_preparation_v1','mode':'pts','scope':'fixture','results':[prepared]})
        roster=self.root/'roster.json';save_new(roster,{'episodes':[row]})
        result=fresh.initialize(self.paths,index,roster,fresh.SOURCE/'manifests/external_assets.json',scope='fixture',minutes=150)
        return Path(result['root']),result
    def test_fresh_init_bridge_empty_parent_and_no_calls(self):
        root,result=self.fixture_run();cfg,session=fresh.runtime_config(root)
        state,journal=fresh.validate_budget(root,cfg);self.assertEqual(state['baseline'],{'source':None});self.assertEqual(journal['calls'],0)
        self.assertFalse((root/'budget/calls.jsonl').exists());ep=load(root/'inputs/episode_000000.json');pts=load(st.verify_ref(ep['whole_pts']))
        self.assertEqual(ep['sources'],[[0,1],[3]]);self.assertEqual(ep['checker_hashes'],pts['code_hashes']);self.assertIn('source_preparation_ref',pts)
        self.assertIsNone(result['prediction_producer_commit']);self.assertEqual(session['mode'],'fresh_reproduction');self.assertNotIn('execution_session',cfg)
    def test_renew_preserves_fresh_counts_and_unknown_refuses(self):
        root,_=self.fixture_run();cfg,_=fresh.runtime_config(root);ledger=fresh.FreshLedger(root,cfg)
        ledger.invoke('detector',lambda:7,episode=0,frame=0,attempt='fixture_only',stage='fixture',n=3)
        before,_=fresh.validate_budget(root,cfg);fresh.renew(root,150);newcfg,_=fresh.runtime_config(root);after,_=fresh.validate_budget(root,newcfg)
        self.assertEqual(before['counts'],after['counts']);self.assertEqual(after['counts'],{'detector':1});self.assertEqual(after['baseline'],{'source':None})
        with ledger.edit() as state:state['pending']['unknown_fixture']={'unknown':True}
        with self.assertRaises(st.BudgetStop):fresh.renew(root,150)
    def test_expired_session_requires_explicit_renewal(self):
        root,_=self.fixture_run();session=load(st.verify_ref(load(root/'session_latest.json')));session['monotonic_deadlines']['hard_stop']=time.monotonic()-1
        target=root/'sessions/expired_fixture.json';save_new(target,session);st.atomic_json(root/'session_latest.json',ref(target))
        with self.assertRaises(ContractError):fresh.runtime_config(root)
        fresh.renew(root,150);self.assertGreater(fresh.runtime_config(root)[1]['monotonic_deadlines']['hard_stop'],time.monotonic())
    def test_fixture_controller_real_owned_child_has_no_models(self):
        root,_=self.fixture_run();result=fresh.execute(root,[0],fixture=True)
        self.assertEqual(result['mode'],'fixture');self.assertIsNone(result['producer_commit'])
        receipt=load(st.verify_ref(load(root/'episode_runs/episode_000000/fixture_latest.json')))
        self.assertTrue(receipt['details']['fixture']);self.assertEqual(receipt['details']['real_model_calls'],0)
        self.assertIsNone(receipt['thread_observation']['opencv_reported_threads']);self.assertFalse((root/'budget/calls.jsonl').exists())
        fresh.assert_closed(root)
        with self.assertRaises(ContractError):fresh.execute(root,[0],allow_models=True,fixture=True)
        with self.assertRaises(ContractError):fresh.export(root)
    def test_production_ancestor_and_unleased_worker_refused(self):
        production=self.root/'fake_production';production.mkdir();save_new(production/'run_manifest.json',{'schema':'r12_run_v1'})
        with self.assertRaises(ContractError):fresh.reject_historical_ancestors(production/'fresh_runs/new')
        root,_=self.fixture_run();man=fresh.verify_fresh_root(root);cfg,_=fresh.runtime_config(root)
        bad=self.root/'lease.json';save_new(bad,{'schema':'invalid'})
        with self.assertRaises(ContractError):validate_lease(root,man,cfg,0,'A',bad,sha(bad))
    def test_dense_extraction_matches_preserved_C3_core(self):
        plan={'fixture':'synthetic B plan'};br=self.root/'B';br.mkdir();planref=save_new(br/'plan.json',plan)
        identity=br/'identity.npz';np.savez_compressed(identity,scales=np.arange(68,dtype=np.float32)*.01,shape=np.arange(45,dtype=np.float32)*.001)
        mesh=br/'mesh.npz';np.savez_compressed(mesh,vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),faces=np.array([[0,1,2]]))
        for fid in (0,2):
            output=br/f'{fid}.npz';np.savez_compressed(output,adapted_params=np.arange(204)*.001+fid,R=np.eye(3),t=np.array([fid,.2,1.]))
            save_new(br/'frames'/f'{fid:06d}.json',{'plan_hash':digest(plan),'output':ref(output)})
        b=self.root/'B_receipt.json';save_new(b,{'status':'completed','details':{'plan':planref,'identity':ref(identity),'mesh':ref(mesh)}})
        class Ledger:
            def assert_no_unknown(self):return {}
            def reserve_bytes(self,*args,**kwargs):pass
        ep={'episode':0,'required':[0,2]};man={'code':{'commit':'0'*40},'audit':{'pairs':[],'full_mesh_pairs':[]}};ledger=Ledger()
        def render(er,ep,man,br,writer):writer.json(er/'quality_audit.json',{'fixture':True});return []
        def pack(root,cfg,man,coverage,inputs,writer,label):writer.json(root/'packs'/label/'validation.json',{'fixture':True});return {}
        roots=[]
        for name,function in [('original',r11_pack.stage_c),('fresh',dense_stage)]:
            root=self.root/name;er=root/'episode_runs/episode_000000';save_new(er/'B_latest.json',ref(b));writer=st.Writer(ledger)
            with patch.object(r11_pack,'render_audit',render),patch.object(r11_pack,'pack_local',pack):function(root,er,ep,{},man,ledger,writer,None)
            roots.append(root)
        for logical in ('episodes/episode_000000.npz','masks/episode_000000.npz'):
            with np.load(roots[0]/logical) as a,np.load(roots[1]/logical) as b:
                self.assertEqual(a.files,b.files)
                for key in a.files:self.assertEqual(a[key].dtype,b[key].dtype);self.assertTrue(np.array_equal(a[key],b[key],equal_nan=True))
        self.assertEqual((roots[0]/'episodes/episode_000000_object.obj').read_bytes(),(roots[1]/'episodes/episode_000000_object.obj').read_bytes())

if __name__=='__main__':unittest.main(verbosity=2)
