"""Synthetic stage and recovery integration. Stub networks only; no real model forward/decode."""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from types import SimpleNamespace
import argparse,copy,datetime,json,os,tempfile,time,traceback
import numpy as np
from common import load,sha,nearest,proxy_geometry
from r11_geometry import complete_proxy_source as complete_proxy
from tracking import digest
from r11_state import *
from r11_inputs import audit_set,segments
from r11_worker import stage_a,stage_b,EXPECTED,restore_record
from r11_pack import stage_c,check_dense,choose_sample,pack_local

CASES=[]
def case(name,fn):
    t=time.monotonic();fn();CASES.append({'name':name,'status':'passed','seconds':time.monotonic()-t})
def reject(fn,typ=Exception):
    try:fn()
    except typ:return
    raise AssertionError('expected rejection')

def main(output,failure_mode='exception'):
    import cv2,av,onnxruntime as ort,torch
    from vendor.nanodet import NanoDet
    base=Path(__file__).parent;cfg=load(base/'config_r11_20261017.json');cfg=copy.deepcopy(cfg)
    future=(datetime.datetime.now(TZ)+datetime.timedelta(days=1)).isoformat()
    cfg['deadlines']={k:future for k in cfg['deadlines']};cfg['call_limits']={k:100 for k in cfg['call_limits']}
    with tempfile.TemporaryDirectory(prefix='r11_synthetic_') as td:
        root=Path(td);epdir=root/'episode_runs/episode_000016';ledger=Ledger(root,cfg);writer=Writer(ledger)
        video=root/'synthetic_video_identity.bin';video.write_bytes(b'no video: synthetic frames supplied by stub')
        mappings=[{'original_frame':i,'pts':512*i} for i in range(53)];stream={'index':0,'time_base':'1/15360','width':32,'height':24,'codec_name':'synthetic'}
        ident={'source_repository':'synthetic','revision':'synthetic','repository_path':'synthetic','declared_size_bytes':video.stat().st_size,'expected_sha256':sha(video)};pts={'frame_map':mappings,'identity':ident,'stream':stream,'code_hashes':{'synthetic':'0'*64}}
        ptspath=root/'synthetic_pts.json';atomic_json(ptspath,pts)
        ep={'schema':'r11_episode_input_v1','episode':16,'source_identity':ident,'video':ref(video),'whole_pts':ref(ptspath),'checker_hashes':pts['code_hashes'],'required':[50,51,52],'sources':[[50,51,52]],'source_map':{str(f):0 for f in (50,51,52)},'stream':stream,'decoded_frame_count':53}
        synthetic_roster=root/'synthetic_roster.json';atomic_json(synthetic_roster,{'episodes':[{'episode':16,**ident,'required_original_frames':[50,51,52],'required_frame_stretches':[{'first':50,'last':52,'count':3}]}]});cfg['assets']['roster']=ref(synthetic_roster)
        man={'code':{'commit':'e'*40,'tree':'f'*40},'audit':{'pairs':[[16,f] for f in (50,51,52)],'full_mesh_pairs':[[16,50]]}}
        fake_counts={'dnn_load':0,'detector':0,'ort_load':0,'hmr':0,'rig_load':0,'rig':0}
        dr=[np.zeros(s,np.float32) for s in ((1,2704,80),(1,2704,32),(1,676,80),(1,676,32),(1,169,80),(1,169,32))]
        class FakeNet:
            def getUnconnectedOutLayersNames(self):return [f'output_{i}' for i in range(6)]
            def setInput(self,b):pass
            def forward(self,n):fake_counts['detector']+=1;return dr
        class FakeDetector(NanoDet):
            def __init__(self,*args):fake_counts['dnn_load']+=1;self.net=FakeNet()
            def post_process(self,raw):return np.array([[2.,2.,25.,22.,.9,0.]])
        class Frame:
            width=32;height=24;time_base=__import__('fractions').Fraction(1,15360)
            def __init__(self,f):self.pts=f*512;self.f=f
            def to_ndarray(self,format):return np.full((24,32,3),self.f,np.uint8)
        class Container:
            def __init__(self):self.streams=SimpleNamespace(video=[SimpleNamespace(index=0,time_base=__import__('fractions').Fraction(1,15360),codec_context=SimpleNamespace(name='synthetic'))])
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def decode(self,stream):return iter(Frame(i) for i in range(53))
        raw={k:np.zeros(v,np.float32) for k,v in EXPECTED.items()};raw['joints_3d'][0]=np.random.default_rng(10).normal(0,.1,(70,3));raw['joints_3d'][0,:,2]+=2;raw['cam_trans'][0]=[.1,.2,1.]
        class FakeSession:
            def __init__(self,*args,**kw):fake_counts['ort_load']+=1
            def get_providers(self):return ['CPUExecutionProvider']
            def get_inputs(self):return [SimpleNamespace(name=k,type='tensor(float)',shape=list(v)) for k,v in {'image':(1,3,224,224),'cliff_cond':(1,3)}.items()]
            def get_outputs(self):return [SimpleNamespace(name=k) for k in EXPECTED]
            def run(self,names,inputs):
                fake_counts['hmr']+=1
                if fake_counts['hmr']==2:
                    if failure_mode=='exception':raise RuntimeError('synthetic HMR known failure')
                    bad={k:v.copy() for k,v in raw.items()}
                    if failure_mode=='nonfinite':bad['joints_3d'][0,0,0]=np.nan
                    elif failure_mode=='shape':bad['joints_2d']=bad['joints_2d'][:,:69]
                    return [bad[k] for k in names]
                return [raw[k].copy() for k in names]
        original_commit=FrameStore.commit
        def crash_commit(store,record):
            original_commit(store,record)
            if record['frame']==51:raise BudgetStop('synthetic postcommit crash')
        with patch('vendor.nanodet.NanoDet',FakeDetector),patch.object(av,'open',lambda *a,**k:Container()),patch.object(ort,'InferenceSession',FakeSession):
            def interrupted():
                with Owner(epdir/'owners/A','A',16) as owner,patch.object(FrameStore,'commit',crash_commit):
                    reject(lambda:stage_a(root,epdir,ep,cfg,man,ledger,writer,owner),BudgetStop)
                r=FrameStore(epdir,writer).records();assert r[51]['hmr_outcome']=='hmr_failed' and r[50]['hmr_outcome']=='hmr_valid';assert fake_counts['hmr']==2
            case('A_closed_failure_commit_crash',interrupted)
            def resume():
                with Owner(epdir/'owners/A','A',16) as owner:r=stage_a(root,epdir,ep,cfg,man,ledger,writer,owner)
                assert r['hmr_valid']==[50,52] and r['hmr_failed']==[51] and fake_counts['hmr']==3 and fake_counts['detector']==3
                atomic_json(epdir/'A_synthetic.json',{'status':'completed','details':r})
            case('A_sameC_resume_no_failed_forward_repeat',resume)
            def already():
                before=dict(fake_counts)
                with Owner(epdir/'owners/A','A',16) as owner:stage_a(root,epdir,ep,cfg,man,ledger,writer,owner)
                assert before==fake_counts
            case('A_all_committed_revalidation_zero_load_or_forward',already)
            def changed_rgb():
                original=Frame.to_ndarray
                with patch.object(Frame,'to_ndarray',lambda self,format:np.zeros((24,32,3),np.uint8)):
                    with Owner(epdir/'owners/A','A',16) as owner:reject(lambda:stage_a(root,epdir,ep,cfg,man,ledger,writer,owner))
            case('A_RGB_parent_change_rejected',changed_rgb)
        class FakeRig:
            forward=SimpleNamespace(schema=SimpleNamespace(arguments=[SimpleNamespace(name=x) for x in ('self','identity_coeffs','model_parameters','face_expr_coeffs','apply_correctives')]))
            def eval(self):return self
            def __call__(self,ti,p,expr,correctives):
                fake_counts['rig']+=1;xyz=(p[0,:3]*10).numpy();v=np.zeros((1,18439,3),np.float32)+xyz;j=np.zeros((1,127,8),np.float32);j[0,:,:3]=xyz
                return torch.tensor(v),torch.tensor(j)
        def fake_load(*a,**kw):fake_counts['rig_load']+=1;return FakeRig()
        # Thread setup is process-once in production; bypass it for repeated in-process tests.
        with patch.object(torch.jit,'load',fake_load),patch.object(torch,'set_num_interop_threads',lambda *a:None):
            def interrupted_rig():
                original_json=writer.json
                def stop_baseline(path,value,*args,**kw):
                    result=original_json(path,value,*args,**kw)
                    if str(path).endswith('000050_baseline.json'):raise BudgetStop('synthetic rig baseline safe-point crash')
                    return result
                with Owner(epdir/'owners/B','B',16) as owner,patch.object(writer,'json',stop_baseline):reject(lambda:stage_b(root,epdir,ep,cfg,man,ledger,writer,owner),BudgetStop)
                assert fake_counts['rig']==1
            case('B_interruption_after_durable_baseline_preserves_forward',interrupted_rig)
            def geometry():
                with Owner(epdir/'owners/B','B',16) as owner:r=stage_b(root,epdir,ep,cfg,man,ledger,writer,owner)
                assert r['hmr_filled']==[51] and fake_counts['rig']==6
                if failure_mode!='exception':assert 51 in r['identity_samples']
                receipt=epdir/'B_synthetic.json';atomic_json(receipt,{'status':'completed','details':r});atomic_json(epdir/'B_latest.json',ref(receipt))
            case('B_episode_identity_same_source_fill_two_full_rig_per_frame',geometry)
            def failed_source():
                gap=copy.deepcopy(ep);gap['sources']=[[50],[51],[52]];gap['source_map']={'50':0,'51':1,'52':2};before=dict(fake_counts)
                with Owner(epdir/'owners/B','B',16) as owner:
                    try:stage_b(root,epdir,gap,cfg,man,ledger,writer,owner)
                    except ValueError as e:assert 'source has no valid HMR parent' in str(e)
                    else:raise AssertionError('failed source borrowed a parent across gap')
                assert before==fake_counts
            case('B_all_failed_source_refuses_cross_gap_fill',failed_source)
            def geometry_resume():
                before=dict(fake_counts)
                with Owner(epdir/'owners/B','B',16) as owner:stage_b(root,epdir,ep,cfg,man,ledger,writer,owner)
                assert before==fake_counts
            case('B_committed_resume_zero_rig_load_or_forward',geometry_resume)
        def localpack():
            with Owner(epdir/'owners/C','C',16) as owner:r=stage_c(root,epdir,ep,cfg,man,ledger,writer,owner)
            pack=load(r['pack']['path']);assert pack['rows']==8231+3*50 and pack['coverage']=='partial' and not pack['submission_eligible'];assert len(pack['per_episode'])==1
        case('C_dense_mask_official_assembly_local_writer_roundtrip',localpack)
        def multiple_episodes():
            official=load(load(base/'config_r11_20261017.json')['assets']['roster']['path']);ids0=official['episodes'][0]['required_original_frames'][:2]
            d=read_npz(root/'episodes/episode_000016.npz');mesh=load(epdir/'B_synthetic.json')['details']['mesh'];original=load(root/'packs/episode_000016/validation.json')['inputs']['16'];T=max(ids0)+1
            zero={'pose':np.full((T,136),np.nan),'scales':np.ones(68,np.float32),'shape':np.ones(45,np.float32),'object_rotation':np.full((T,3,3),np.nan),'object_translation':np.full((T,3),np.nan),'object_scale':np.array(1.)}
            for f in ids0:zero['pose'][f]=d['pose'][50]+1;zero['object_rotation'][f]=d['object_rotation'][50];zero['object_translation'][f]=d['object_translation'][50]+1
            nr=writer.npz(root/'episodes/episode_000000.npz','C',**zero);obj=root/'episodes/episode_000000_object.obj';writer.bytes(obj,(root/'episodes/episode_000016_object.obj').read_bytes(),'C')
            inputs={16:original,0:{'npz':nr,'obj':ref(obj),'mesh':mesh}}
            result=pack_local(root,cfg,man,{16:[50,51,52],0:ids0},inputs,writer,'synthetic_two_episodes');assert result['rows']==2*8231+5*50 and len(result['per_episode'])==2
            previous=ref(root/'packs/synthetic_two_episodes/validation.json');again=pack_local(root,cfg,man,{16:[50,51,52],0:ids0},inputs,writer,'synthetic_two_episodes');assert previous==ref(root/'packs/synthetic_two_episodes/validation.json') and again==result
        case('C_two_episodes_independent_arrays_and_exact_global_sample_order',multiple_episodes)
        def failed_raw_recovery():
            from r11_worker import valid_raw
            records=FrameStore(epdir,writer).records();record=copy.deepcopy(records[52]);bad={k:v.copy() for k,v in raw.items()};bad['joints_3d'][0,0,0]=np.nan
            result,end=ledger.invoke('hmr',lambda:bad,episode=18,frame=52,attempt='synthetic',stage='A',n=3,output_identity=arrays_hash)
            assert not valid_raw(result);assert read_line_ref(end)['outcome']=='returned'
            wrong={k:v.copy() for k,v in raw.items()};wrong['mhr_params']=wrong['mhr_params'][:,:203];assert not valid_raw(wrong)
            assert not valid_raw({k:v.astype(np.float64) for k,v in raw.items()})
        case('HMR_wrong_shape_dtype_nonfinite_are_known_failed_not_valid',failed_raw_recovery)
        def invalid_dense():
            d=read_npz(root/'episodes/episode_000016.npz');mask=read_npz(root/'masks/episode_000016.npz')['mask'];bad=copy.deepcopy(d);bad['object_translation']=bad['object_translation'][:-1];reject(lambda:check_dense(bad,[50,51,52],mask));wrong=mask.copy();wrong[0]=True;reject(lambda:check_dense(d,[50,51,52],wrong))
        case('C_independent_translation_T_and_exact_mask_rejection',invalid_dense)
        def source_isolation():
            assert nearest(2,[1,3])==1
            g={1:proxy_geometry(raw),3:proxy_geometry(raw)};g[3]['R']=None
            _,_,one=complete_proxy({3:g[3]},dimensions=np.ones(3));assert one[3]['rotation_parent'] is None
        case('GEO_nearest_tie_and_source_local_no_foreign_parent',source_isolation)
        def active_owner():
            with Owner(root/'active','A',1):reject(lambda:Owner(root/'active','A',1).__enter__())
        case('owner_active_lock_rejected_history_retained',active_owner)
        def unknown():
            try:ledger.invoke('hmr',lambda:(_ for _ in ()).throw(BudgetStop('simulated kill')),episode=7,frame=1,attempt='synthetic',stage='A',n=2)
            except BudgetStop:pass
            assert len(ledger.unknown_for(7,1))==1
            assert all(x['frame']!=51 for x in ledger.unknown_for(16))
            try:ledger.invoke('hmr',lambda:(_ for _ in ()).throw(BudgetStop('synthetic end/state crash')),episode=8,frame=1,attempt='synthetic',stage='A',n=2)
            except BudgetStop:pass
            pending=ledger.snapshot()['pending'];cid,start=next((k,v) for k,v in pending.items() if v['episode']==8)
            append_line(ledger.journal,{**start,'kind':'call_end','outcome':'failed','error':'synthetic closed exception','time':now(),'elapsed_s':0.,'output_identity':None})
            assert not ledger.unknown_for(8,1) and len(ledger.unknown_for(7,1))==1
        case('call_start_without_end_unknown_charged',unknown)
        def budgets():
            tiny=copy.deepcopy(cfg);tiny['storage_limits']['small']=8;l=Ledger(root/'tiny',tiny);w=Writer(l);w.bytes(root/'tiny/a',b'12345678');reject(lambda:w.bytes(root/'tiny/b',b'x'),BudgetStop);w.remove(root/'tiny/a','small');w.bytes(root/'tiny/b',b'1234')
            stopped=copy.deepcopy(cfg);stopped['deadlines']['candidate_freeze']='2000-01-01T00:00:00+08:00';stopped['deadlines']['candidate_stop']=stopped['deadlines']['candidate_freeze'];reject(lambda:Ledger(root/'stop',stopped).invoke('hmr',lambda:1,episode=0,frame=0,attempt='t',stage='A',n=1),BudgetStop)
        case('byte_budget_releases_transient_and_absolute_stop',budgets)
        def load_budgets():
            c=copy.deepcopy(cfg);c['call_limits']={**c['call_limits'],'dnn_load':35,'ort_session_load':35,'rig_load':35};l=Ledger(root/'loads',c)
            for kind in ('dnn_load','ort_session_load','rig_load'):
                for i in range(35):l.invoke(kind,lambda:None,episode=i,frame=None,attempt='synthetic',stage='A',n=1)
                reject(lambda:l.invoke(kind,lambda:None,episode=99,frame=None,attempt='synthetic',stage='A',n=1),BudgetStop)
            assert sum(l.snapshot()['counts'].values())==105
        case('three_independent_35_load_limits',load_budgets)
        def process_group_stop():
            import subprocess,psutil,sys
            import r11
            c=copy.deepcopy(cfg);c['stage_limits']['A']['wall_s']=.2;c['monitor_interval_s']=.05
            real_popen=subprocess.Popen
            script="import subprocess,sys,time; c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);print(c.pid,flush=True);time.sleep(30)"
            def fake_worker(cmd,**kwargs):return real_popen([sys.executable,'-c',script],**kwargs)
            with patch.object(r11,'verify_code',lambda *a:None),patch.object(r11,'global_new_bytes',lambda *a:0),patch.object(subprocess,'Popen',fake_worker):
                reject(lambda:r11.supervise(root/'supervisor_fixture',c,{'code':{'commit':'e'*40},'kit_source_hashes':{}},'A',16),BudgetStop)
            logs=list((root/'supervisor_fixture/supervision').glob('*.stdout.log'));pid=int(logs[0].read_text().strip())
            time.sleep(.1)
            if psutil.pid_exists(pid):assert psutil.Process(pid).status()==psutil.STATUS_ZOMBIE
        case('supervisor_real_subprocess_group_timeout_no_model',process_group_stop)
        def atomic_index_crash():
            store=FrameStore(epdir,writer);before=store.journal.read_bytes()
            real_replace=os.replace
            def fail_replace(src,dst):
                if Path(dst)==store.journal:raise OSError('synthetic crash before atomic index replace')
                return real_replace(src,dst)
            with patch.object(os,'replace',fail_replace):reject(lambda:atomic_json(store.journal,[]))
            assert store.journal.read_bytes()==before and sorted(store.records())==[50,51,52]
        case('frame_index_atomic_replace_failure_preserves_safe_commit_set',atomic_index_crash)
        def frozen_data_contract():
            from r11_inputs import freeze_episode,verify_episode
            actualcfg=load(base/'config_r11_20261017.json');roster=load(actualcfg['assets']['roster']['path']);frozen=freeze_episode(roster['episodes'][16],actualcfg)
            assert frozen['whole_pts_verified'] and frozen['required']==list(range(50,290))
            bad=copy.deepcopy(frozen);bad['required']=bad['required'][1:];reject(lambda:verify_episode(bad,actualcfg))
        case('actual_T05_frozen_receipt_input_contract_without_decoding',frozen_data_contract)
        def audit_contract():
            roster=load(load(base/'config_r11_20261017.json')['assets']['roster']['path']);a=audit_set(roster);assert len(a['pairs'])==135
            frozen=Path(cfg['root'])/'DD/证据/T011.08/固定135审计对_20261017.json'
            if frozen.exists():
                assert sha(frozen)=='4135ad98260fdaf65721a0d92ff6455f9773ef0c504f37e81042c2e9154eb34f'
                d=load(frozen);pairs=d if isinstance(d,list) else d.get('pairs') or d.get('audit_pairs')
                if pairs is not None:
                    if pairs and isinstance(pairs[0],dict):pairs=[[x['episode'],x.get('original_frame',x.get('frame'))] for x in pairs]
                    assert sorted(pairs)==a['pairs']
            assert len(segments(roster['episodes'][16])[0])==1
        case('fixed_roster_135_audit_and_source_mapping',audit_contract)
        state=ledger.snapshot();result={'schema':'r11_synthetic_checks_v1','status':'passed','date':'2026-10-17','as_of':now(),'cases':CASES,'hmr_failure_mode':failure_mode,'synthetic_stub_calls':fake_counts,'synthetic_ledger_counts':state['counts'],'unknown_preserved':state['pending'],'real_model_loads':0,'real_model_forwards':0,'real_video_decodes':0,'external_actions':0,'source_hashes':{p.name:sha(p) for p in base.glob('r11*.py')},'limitations':['Synthetic stage integration uses explicit network and video stubs.','Real240, full30, physical quality and resource performance not run.']}
        atomic_json(output,result,True);print(json.dumps({'status':'passed','groups':len(CASES),'stub_calls':fake_counts,'output':str(output)},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--failure-mode',choices=['exception','nonfinite','shape'],default='exception');args=ap.parse_args()
    try:main(args.output,args.failure_mode)
    except BaseException as e:
        failure=args.output.with_name(args.output.stem+'_failure.json');atomic_json(failure,{'status':'failed','as_of':now(),'cases':CASES,'error':repr(e),'traceback':traceback.format_exc()},True);raise
