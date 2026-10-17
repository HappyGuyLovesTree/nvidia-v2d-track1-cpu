"""Zero-model / zero-video fixture checks; optional actual stock writer on synthetic data."""
from pathlib import Path
from fractions import Fraction
import ast, copy, contextlib, csv, errno, hashlib, io, json, math, os, stat, subprocess, sys, tempfile, unittest, zipfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v2d_public.common import ContractError, Paths, digest, load, ref, run_logged, save_new, sha
from v2d_public import common
from v2d_public.inputs import validate_pts, validate_roster, required_from_keys, verify_file, official_binding, prepare_one
from v2d_public.assets import verify_asset, download, download_curl, extract_zip
from v2d_public.packing import check_view, main as pack_main, verify_kit

ROOT=Path(__file__).resolve().parents[1]

def fixture_row(payload=b'fixture bytes, not a video'):
    return {'episode':0,'source_repository':'fixture','revision':'fixture','repository_path':'episode_000000.bin',
       'declared_size_bytes':len(payload),'expected_sha256':hashlib.sha256(payload).hexdigest(),
       'declared_frame_count':4,'declared_fps':30,'declared_video':{'video.width':32,'video.height':24,'video.codec':'h264','video.pix_fmt':'yuv420p'},
       'required_original_frames':[0,1,3],'required_frame_stretches':[{'first':0,'last':1,'count':2},{'first':3,'last':3,'count':1}]}

def fixture_probe():
    return {'streams':[{'index':0,'width':32,'height':24,'codec_name':'h264','pix_fmt':'yuv420p','time_base':'1/15360'}],
       'frames':[{'pts':i*512,'best_effort_timestamp':i*512,'pkt_duration':512} for i in range(4)]}

class Base(unittest.TestCase):
    def setUp(self):
        root=os.environ.get('V2D_TEST_OUTPUT')
        if root:
            Path(root).mkdir(parents=True,exist_ok=True)
            self.base=Path(tempfile.mkdtemp(prefix=self._testMethodName+'_',dir=root))
            self.cleanup_fixture=False
        else:
            self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.cleanup_fixture=True
    def tearDown(self):
        if self.cleanup_fixture:self.temp.cleanup()

class InputTests(Base):
    def test_complete_pts_preserves_original_map(self):
        r=validate_pts(fixture_row(),fixture_probe(),0,'')
        self.assertTrue(r['whole_pts_verified']);self.assertEqual([x['original_frame'] for x in r['frame_map']],[0,1,2,3]);self.assertEqual(r['required_covered'],3)
    def test_invalid_pts_variants(self):
        for case in ('duplicate','reverse','missing','boolean','wrong_interval','dimensions','codec','exit','stderr'):
            with self.subTest(case=case):
                row=fixture_row();data=fixture_probe();exitcode=0;stderr=''
                if case=='duplicate':data['frames'][2]['pts']=512
                if case=='reverse':data['frames'][2]['pts']=1
                if case=='missing':data['frames'].pop()
                if case=='boolean':data['frames'][1]['pts']=True
                if case=='wrong_interval':data['frames'][2]['pts']=1023
                if case=='dimensions':data['streams'][0]['width']=99
                if case=='codec':data['streams'][0]['codec_name']='hevc'
                if case=='exit':exitcode=1
                if case=='stderr':stderr='decode error'
                self.assertFalse(validate_pts(row,data,exitcode,stderr)['whole_pts_verified'])
    def test_frozen_checker_equivalence(self):
        path=os.environ.get('V2D_TEST_FROZEN_CHECKER')
        if not path:self.skipTest('pass optional frozen checker source path')
        tree=ast.parse(Path(path).read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_pts')
        namespace={'Fraction':Fraction};exec(compile(ast.Module(body=[function],type_ignores=[]),'<pure frozen validate_pts>','exec'),namespace)
        for case in range(8):
            data=fixture_probe();row=fixture_row()
            if case==1:data['frames'][0]['pts']=None
            if case==2:data['frames'][1]['pts']=True
            if case==3:data['streams'][0]['time_base']='0'
            if case==4:data['frames'].pop()
            if case==5:row['declared_fps']=25
            if case==6:data['streams'][0]['width']=0
            if case==7:data['frames'][2]['pts']=512
            self.assertEqual(validate_pts(row,data,0,''),namespace['validate_pts'](row,data,0,''))
    def test_sample_keys_and_gap_mapping(self):
        rows=['t1_000000_000000_0_000000','t1_000000_000001_3_000000','t1_000000_000003_4_000000','t1_000000_999999_1_000000']
        self.assertEqual(required_from_keys(rows),{0:[0,1,3]})
        with self.assertRaises(ContractError):required_from_keys(rows+rows[:1])
        with self.assertRaises(ContractError):required_from_keys(['t1_000000_999999_0_000000'])
        with self.assertRaises(ContractError):required_from_keys(['t1_000000_000000_7_000000'])
    def test_file_identity_not_official_binding(self):
        path=self.base/'input.bin';path.write_bytes(b'fixture bytes, not a video');row=fixture_row()
        self.assertTrue(verify_file(row,path)['file_verified']);self.assertFalse(official_binding(row,None)['official_independent_binding'])
        path.write_bytes(b'other bytes');self.assertFalse(verify_file(row,path)['file_verified'])
    def test_official_binding_conflicts_and_evidence(self):
        row=fixture_row();evidence=self.base/'official_metadata_fixture.json';evidence.write_text('{}')
        b={k:row[k] for k in ('episode','source_repository','revision','repository_path','declared_size_bytes')}
        b.update(result='passed',lfs_content_sha256=row['expected_sha256'],evidence_refs=[ref(evidence)])
        self.assertTrue(official_binding(row,{'entries':[b]})['official_independent_binding'])
        b['lfs_content_sha256']='0'*64
        with self.assertRaises(ContractError):official_binding(row,{'entries':[b]})
        b['lfs_content_sha256']=row['expected_sha256'];evidence.write_text('changed')
        with self.assertRaises(ContractError):official_binding(row,{'entries':[b]})
    def test_fixture_full_preparation_no_decode(self):
        # The executable emits synthetic ffprobe-shaped JSON; it cannot decode any video.
        executable=self.base/'probe_fixture';executable.write_text('#!'+sys.executable+'\nimport json\nprint('+repr(json.dumps(fixture_probe()))+')\n');executable.chmod(0o700)
        payload=self.base/'fixture.bin';payload.write_bytes(b'fixture bytes, not a video')
        result=prepare_one(fixture_row(),payload,Paths(self.base/'workspace'),ffprobe=executable,scope='fixture')
        self.assertEqual(result['status'],'passed');ep=load(result['episode_input']['path'])
        self.assertEqual(ep['sources'],[[0,1],[3]]);self.assertFalse(ep['official_independent_binding']);self.assertIsNone(ep['actual_prediction_producer_commit']);self.assertFalse(result['submission_ready'])
    def test_failure_before_probe_preserves_receipt(self):
        path=self.base/'bad.bin';path.write_bytes(b'bad')
        result=prepare_one(fixture_row(),path,Paths(self.base/'workspace'),ffprobe='/missing',scope='fixture')
        self.assertEqual(result['status'],'failed');record=load(result['whole_pts']['path']);self.assertNotIn('probe',record);self.assertFalse(record['whole_pts_verified'])

class PathAndProcessTests(Base):
    def test_output_escape_and_asset_override(self):
        paths=Paths(self.base/'work',{'model':str(self.base/'external.bin')})
        for name in ('../outside','/absolute','.'):
            with self.assertRaises(ContractError):paths.output(name)
        paths.workspace.mkdir();(paths.workspace/'escape').symlink_to(self.base)
        with self.assertRaises(ContractError):paths.output('escape/out')
        self.assertEqual(paths.asset('model','assets/model'),self.base/'external.bin')
    def test_timeout_and_output_cap_preserve_terminal_record(self):
        r=run_logged([sys.executable,'-c','import time; time.sleep(10)'],self.base/'timeout',timeout=.1)
        self.assertEqual(r['status'],'failed');self.assertIsNotNone(r['returncode']);self.assertEqual(r['error']['type'],'TimeoutError')
        r=run_logged([sys.executable,'-c','print("x"*4096)'],self.base/'too_much',timeout=2,max_output_bytes=64)
        self.assertEqual(r['status'],'failed');self.assertTrue((self.base/'too_much/finished.json').exists())
    def test_receipts_do_not_overwrite(self):
        save_new(self.base/'receipt.json',{'test':1})
        with self.assertRaises(FileExistsError):save_new(self.base/'receipt.json',{'test':2})

class CleanupFaultTests(Base):
    class Process:
        """No PID, real child, signal or network. All waits require a finite bound."""
        def __init__(self,mode):self.mode=mode;self.calls=[];self.polls=0
        def poll(self):
            self.polls+=1;self.calls.append({'operation':'poll'})
            if self.mode=='poll_unknown' and self.polls>1:raise PermissionError(errno.EPERM,'fixture cleanup poll denied')
            return 0 if self.mode in ('normal_wait_error','success') else None
        def kill(self):
            self.calls.append({'operation':'kill'})
            if self.mode in ('kill_error','kill_error_then_exit'):raise PermissionError(errno.EPERM,'fixture kill denied')
        def wait(self,*,timeout):
            if not math.isfinite(timeout) or timeout<0:raise AssertionError('wait requires finite nonnegative timeout')
            self.calls.append({'operation':'wait','timeout':timeout})
            if self.mode in ('kill_error_then_exit','success'):return 0
            if self.mode=='poll_unknown':raise subprocess.TimeoutExpired('fixture',timeout)
            raise ChildProcessError(errno.ECHILD,'fixture wait failed')

    def test_cleanup_faults_preserve_primary_and_terminal(self):
        for mode in ('kill_error','wait_error','kill_error_then_exit','poll_unknown'):
            with self.subTest(mode=mode):
                process=self.Process(mode);directory=self.base/mode
                with patch.object(common.subprocess,'Popen',return_value=process):
                    result=run_logged(['no-real-process'],directory,timeout=1e-9,cleanup_timeout=.125)
                self.assertEqual(result['status'],'failed');self.assertEqual(result['primary_error']['type'],'TimeoutError')
                self.assertEqual(result['error']['type'],'TimeoutError');self.assertTrue((directory/'finished.json').is_file())
                saved=load(directory/'finished.json');self.assertEqual(saved,result)
                waits=[c for c in process.calls if c['operation']=='wait'];self.assertTrue(waits)
                self.assertTrue(all(0<c['timeout']<=.125 for c in waits))
                if mode in ('kill_error','kill_error_then_exit'):self.assertEqual(result['kill_error']['errno'],errno.EPERM)
                if mode=='kill_error':self.assertEqual(result['wait_errors'][0]['errno'],errno.ECHILD)
                if mode=='kill_error_then_exit':self.assertTrue(result['cleanup']['reaped']);self.assertEqual(result['returncode'],0)
                else:
                    self.assertFalse(result['cleanup']['reaped']);self.assertIsNone(result['stdout']['sha256'])
                if mode=='poll_unknown':
                    self.assertIsNotNone(result['cleanup_poll_error']);self.assertFalse(result['cleanup']['kill_attempted'])
                    self.assertFalse(any(c['operation']=='kill' for c in process.calls))

    def test_normal_wait_failure_is_bounded_and_separate(self):
        process=self.Process('normal_wait_error')
        with patch.object(common.subprocess,'Popen',return_value=process):
            result=run_logged(['no-real-process'],self.base/'normal_wait',timeout=1,cleanup_timeout=.125)
        self.assertEqual(result['primary_error']['errno'],errno.ECHILD)
        self.assertEqual([e['operation'] for e in result['wait_errors']],['normal_wait','cleanup_wait'])
        self.assertEqual(next(c for c in process.calls if c['operation']=='wait')['timeout'],0)
        self.assertFalse(any(c['operation']=='kill' for c in process.calls))

    def test_started_receipt_error_prevents_spawn(self):
        original=common.save_new
        def writer(path,value):
            if Path(path).name=='started.json':raise OSError(errno.ENOSPC,'fixture started receipt disk full')
            return original(path,value)
        with patch.object(common,'save_new',writer),patch.object(common.subprocess,'Popen') as spawn:
            result=run_logged(['no-real-process'],self.base/'start_error',timeout=1)
        spawn.assert_not_called();self.assertEqual(result['primary_error']['errno'],errno.ENOSPC)
        self.assertEqual(result['receipt_errors'][0]['operation'],'started_receipt_write');self.assertEqual(result['cleanup']['status'],'not_spawned')
        self.assertTrue((self.base/'start_error/finished.json').is_file())

    def test_terminal_and_fallback_receipt_errors_independent(self):
        original=common.save_new
        for unavailable in (False,True):
            directory=self.base/str(unavailable);process=self.Process('kill_error')
            def writer(path,value):
                name=Path(path).name
                if name=='finished.json' or (unavailable and name.startswith('finished_fallback_')):
                    raise OSError(errno.ENOSPC,'fixture terminal receipt full')
                return original(path,value)
            class BrokenStderr:
                def write(self,value):raise OSError(errno.EIO,'fixture stderr unavailable')
                def flush(self):raise OSError(errno.EIO,'fixture stderr unavailable')
            with patch.object(common,'save_new',writer),patch.object(common.subprocess,'Popen',return_value=process),patch.object(common.sys,'stderr',BrokenStderr()):
                result=run_logged(['no-real-process'],directory,timeout=1e-9,cleanup_timeout=.125)
            self.assertEqual(result['primary_error']['type'],'TimeoutError');self.assertEqual(result['kill_error']['errno'],errno.EPERM);self.assertTrue(result['wait_errors'])
            operations=[e['operation'] for e in result['receipt_errors']];self.assertIn('finished_receipt_write',operations)
            if unavailable:
                self.assertEqual(result['terminal_receipt']['status'],'unavailable');self.assertIn('fallback_receipt_write',operations);self.assertIn('stderr_fallback',operations)
            else:
                self.assertEqual(result['terminal_receipt']['status'],'fallback_written');self.assertEqual(load(result['terminal_receipt']['path']),result)

    def test_log_reference_error_does_not_erase_exit(self):
        original=common.ref;process=self.Process('success')
        def broken_ref(path):
            if Path(path).name=='stdout':raise OSError(errno.EIO,'fixture log hash unreadable')
            return original(path)
        with patch.object(common,'ref',broken_ref),patch.object(common.subprocess,'Popen',return_value=process):
            result=run_logged(['no-real-process'],self.base/'reference',timeout=1)
        self.assertEqual(result['returncode'],0);self.assertTrue(result['cleanup']['reaped']);self.assertIsNone(result['primary_error'])
        self.assertEqual(result['receipt_errors'][0]['operation'],'stdout_reference');self.assertEqual(result['status'],'failed')

    def test_cleanup_budget_exhaustion_never_calls_unbounded_wait(self):
        process=self.Process('wait_error');moments=iter([0.,1.,2.,3.,4.,5.])
        with patch.object(common.subprocess,'Popen',return_value=process),patch.object(common.time,'monotonic',lambda:next(moments)):
            result=run_logged(['no-real-process'],self.base/'budget',timeout=.1,cleanup_timeout=.125)
        self.assertEqual(result['primary_error']['type'],'TimeoutError');self.assertEqual(result['wait_errors'][0]['operation'],'cleanup_wait_budget')
        self.assertFalse(result['cleanup']['wait_attempted']);self.assertFalse(any(c['operation']=='wait' for c in process.calls))

class AssetTests(Base):
    def asset(self,payload=b'fixed fixture'):
        return {'key':'test','url':'https://example.invalid/fixed','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
    def opener(self,payload):
        class Response(io.BytesIO):
            status=200;headers={}
            def geturl(self):return 'https://example.invalid/resolved'
        return lambda *args,**kwargs:Response(payload)
    def test_download_one_request_hash_and_no_overwrite(self):
        asset=self.asset();p=self.base/'download.bin';r=download(asset,p,opener=self.opener(b'fixed fixture'))
        self.assertEqual(r['attempts'],1);self.assertTrue(verify_asset(asset,p)['verified'])
        with self.assertRaises(ContractError):download(asset,p,opener=self.opener(b'fixed fixture'))
    def test_download_hash_size_or_scheme_rejected(self):
        for index,payload in enumerate((b'wrong fixture',b'longer than fixed fixture',b'short')):
            with self.assertRaises(ContractError):download(self.asset(),self.base/str(index),opener=self.opener(payload))
            self.assertFalse((self.base/str(index)).exists())
        asset=self.asset();asset['url']='http://example.invalid/fixed'
        with self.assertRaises(ContractError):download(asset,self.base/'http',opener=self.opener(b'fixed fixture'))
    def test_cli_curl_transport_uses_fixed_secure_arguments(self):
        def mocked_run(argv,directory,**limits):
            self.assertEqual(argv[1],'-q');self.assertIn('--proto-redir',argv);self.assertIn('--max-filesize',argv);self.assertEqual(argv[argv.index('--retry')+1],'0')
            self.assertNotIn('--insecure',argv)
            Path(argv[argv.index('--output')+1]).write_bytes(b'fixed fixture')
            directory=Path(directory);directory.mkdir();out=directory/'stdout';out.write_text('200')
            return {'status':'completed','stdout':ref(out)}
        with patch('v2d_public.assets.run_logged',mocked_run):
            result=download_curl(self.asset(),self.base/'download',self.base/'curl_logs',timeout=5)
        self.assertEqual(result['status'],'verified');self.assertFalse(result['model_loaded'])
    def test_zip_only_expected_member_and_content(self):
        p=self.base/'archive.zip'
        with zipfile.ZipFile(p,'w') as z:z.writestr('assets/model.bin',b'fixed fixture');z.writestr('unwanted.bin',b'other')
        target=self.base/'selected.bin';result=extract_zip(p,target,archive_sha256=sha(p),member='assets/model.bin',member_sha256=self.asset()['sha256'],member_bytes=13,max_bytes=13)
        self.assertEqual(target.read_bytes(),b'fixed fixture');self.assertEqual(result['files'],['asset']);self.assertFalse((self.base/'unwanted.bin').exists())
    def test_zip_traversal_symlink_and_cap_rejected(self):
        for index,name,link in [(0,'../outside',False),(1,'link',True),(2,'big',False)]:
            p=self.base/f'{index}.zip'
            with zipfile.ZipFile(p,'w') as z:
                info=zipfile.ZipInfo(name)
                if link:info.external_attr=(stat.S_IFLNK|0o777)<<16
                z.writestr(info,b'1234')
            with self.assertRaises(ContractError):extract_zip(p,self.base/f'out{index}',archive_sha256=sha(p),max_bytes=1 if index==2 else 16)
            self.assertFalse((self.base/f'out{index}').exists())

class StockFixtureTests(Base):
    def test_bad_dense_values_masks_and_ambiguous_mesh_rejected(self):
        import numpy as np
        view=self.base/'view';view.mkdir();masks=self.base/'masks';masks.mkdir();obj=view/'episode_000000_object.obj';obj.write_text('v 0 0 0\n')
        rows=['t1_000000_000001_0_000000']
        good={'pose':np.full((2,136),np.nan),'scales':np.ones(68),'shape':np.zeros(45),'object_rotation':np.full((2,3,3),np.nan),'object_translation':np.full((2,3),np.nan),'object_scale':np.array(1.)}
        good['pose'][1]=0;good['object_rotation'][1]=np.eye(3);good['object_translation'][1]=[0,0,1]
        for case in ('nan_required','rotation','scale','mask','gap_value','mesh'):
            with self.subTest(case=case):
                data={k:v.copy() for k,v in good.items()};mask=np.array([False,True])
                if case=='nan_required':data['pose'][1,0]=np.nan
                if case=='rotation':data['object_rotation'][1]=np.zeros((3,3))
                if case=='scale':data['object_scale']=np.array(-1.)
                if case=='mask':mask=np.array([True,True])
                if case=='gap_value':data['pose'][0]=0
                alternate=obj.with_suffix('.ply')
                if case=='mesh':alternate.write_text('ambiguous')
                np.savez(view/'episode_000000.npz',**data);np.savez(masks/'episode_000000.npz',mask=mask,original_frames=np.array([1]))
                with self.assertRaises(ContractError):check_view(view,rows,masks=masks,scope='fixture')
                if alternate.exists():alternate.unlink()
    def test_actual_unchanged_stock_writer_on_synthetic_arrays(self):
        kit=os.environ.get('V2D_TEST_KIT')
        if not kit:self.skipTest('pass V2D_TEST_KIT for the real fixed stock-writer fixture')
        import numpy as np
        frames=[2];view=self.base/'view';view.mkdir();maskroot=self.base/'masks';maskroot.mkdir()
        pose=np.full((3,136),np.nan);pose[2]=np.arange(136)*.001
        rotations=np.full((3,3,3),np.nan);rotations[2]=np.eye(3)
        translations=np.full((3,3),np.nan);translations[2]=[.1,.2,1]
        np.savez(view/'episode_000000.npz',pose=pose,scales=np.ones(68),shape=np.zeros(45),object_rotation=rotations,object_translation=translations,object_scale=np.array(1.))
        np.savez(maskroot/'episode_000000.npz',mask=np.array([False,False,True]),original_frames=np.array([2]))
        verts=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
        faces=[[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[3,7,6],[3,6,2],[0,4,7],[0,7,3],[1,2,6],[1,6,5]]
        obj=''.join('v '+' '.join(str(v*.1) for v in row)+'\n' for row in verts)+''.join('f '+' '.join(str(i+1) for i in row)+'\n' for row in faces)
        (view/'episode_000000_object.obj').write_text(obj)
        budgets=[(2,0,46),(999999,1,23),(999999,2,15),(2,3,3),(2,4,1),(999999,5,1),(999999,6,4096),(999999,7,4096)]
        rows=[f't1_000000_{fid:06d}_{a}_{point:06d}' for fid,a,n in budgets for point in range(n)]
        sample=self.base/'sample.csv'
        with sample.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['row_id']);writer.writerows([[r] for r in rows])
        self.assertEqual(len(rows),8281)
        before={x.name:sha(x) for x in view.iterdir()}
        paths=self.base/'paths.json';save_new(paths,{'schema':'r12_public_paths_v1','workspace':str(self.base/'workspace'),'asset_overrides':{'kit':kit},'python_extra_sites':os.environ.get('V2D_TEST_EXTRA_SITE','').split(os.pathsep) if os.environ.get('V2D_TEST_EXTRA_SITE') else []})
        commit='https://github.com/HappyGuyLovesTree/nvidia-v2d-track1-cpu/commit/483f8d61690d7df229c57dca43750b23afcfbf57'
        output=io.StringIO()
        with contextlib.redirect_stdout(output):status=pack_main(['run','--paths',str(paths),'--episodes',str(view),'--masks',str(maskroot),'--sample',str(sample),'--commit',commit,'--kit-manifest',str(ROOT/'manifests/stock_kit.json'),'--scope','fixture'])
        result=json.loads(output.getvalue());self.assertEqual(status,0,result);self.assertEqual(result['validation']['rows'],8281);self.assertTrue(result['official_writer_executed']);self.assertFalse(result['submission_ready']);self.assertIsNone(result['pack_producer_commit']);self.assertEqual(before,{x.name:sha(x) for x in view.iterdir()})
        # Reject provenance-less official runs, including a fixture supplied as an official sample.
        with self.assertRaises(ContractError):check_view(view,rows,masks=maskroot,scope='official',build_commit='0'*40)

class ZeroModelTests(unittest.TestCase):
    def test_model_runtime_modules_never_imported_by_test_parent(self):
        self.assertFalse({'torch','onnxruntime','cv2','av'}.intersection(sys.modules))

if __name__=='__main__':unittest.main(verbosity=2)
