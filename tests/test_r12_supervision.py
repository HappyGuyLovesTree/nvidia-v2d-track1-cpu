"""SUP01--04 and RUN01 regressions; mock faults and real OS evidence stay separate.

Run with the frozen Python: python -B tests/test_r12_supervision.py --output DIR
No worker/model/decoder imports are permitted. Fixture descendants self-expire.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse
import copy
import datetime
import errno
import io
import json
import os
import signal
import subprocess
import sys
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import psutil
import r12_supervision as sup


OUT = None
RESULTS = []
REAL = {'supervised_Popen': 0, 'fixture_driver_Popen': 0, 'signals': [], 'created_identities': []}
AUDIT = {'forbidden': []}


def guard(event, args):
    if event == 'import' and args[0].split('.')[0] in ('torch', 'cv2', 'av', 'onnxruntime'):
        AUDIT['forbidden'].append({'event': event, 'name': args[0]})
        raise AssertionError('model/decoder import forbidden')
    if event in ('socket.connect', 'socket.bind', 'os.killpg', 'os.system'):
        AUDIT['forbidden'].append({'event': event})
        raise AssertionError('external/group operation forbidden')
sys.addaudithook(guard)


def session():
    return {'session_id': 'T012.04-fixture',
            'hard_stop': (datetime.datetime.now(sup.TZ)+datetime.timedelta(seconds=30)).isoformat(),
            'cleanup': {'exit_grace_s': .35, 'term_wait_s': .35, 'kill_wait_s': .35,
                        'cleanup_total_s': 1.5, 'telemetry_retry_count': 2,
                        'telemetry_retry_total_s': .2}}


class FakeProc:
    pid = 900001
    def __init__(self, backend):
        self.backend, self.returncode = backend, None
    def poll(self):
        if self.backend.clock >= self.backend.exit_at:
            self.returncode = self.backend.exit_code
        return self.returncode
    def wait(self, timeout):
        self.backend.waits.append(timeout)
        if self.backend.wait_error:
            raise self.backend.wait_error
        if self.poll() is not None:
            return self.returncode
        if self.backend.exit_at <= self.backend.clock + timeout:
            self.backend.clock = self.backend.exit_at
            return self.poll()
        self.backend.clock += timeout
        raise subprocess.TimeoutExpired(['fake-worker'], timeout)


class FakeBackend:
    def __init__(self, *, exit_at=.08, ignore_term=False, stubborn=False):
        self.clock, self.base = 0., time.time()
        self.exit_at, self.exit_code = exit_at, 0
        self.ignore_term, self.stubborn = ignore_term, stubborn
        self.signals, self.waits, self.identities = [], [], []
        self.spawn_count = 0
        self.proc = FakeProc(self)
        self.identity_hook = None
        self.children_hook = None
        self.rss_hook = None
        self.signal_fault = {}
        self.wait_error = None
        self.on_spawn = None
        self.child_ids = []
        self.nidentity = 0
    def monotonic(self):return self.clock
    def wall_time(self):return self.base+self.clock
    def sleep(self, value):self.clock += value
    def spawn(self, *args, **kwargs):
        if self.on_spawn:self.on_spawn()
        self.spawn_count += 1
        return self.proc
    def identity(self, pid, owner_id, attempt_id):
        self.nidentity += 1
        row = {'pid': pid, 'create_time': 100., 'ppid': os.getpid() if pid == self.proc.pid else self.proc.pid,
               'uid': os.getuid(), 'pgid': self.proc.pid, 'sid': self.proc.pid,
               'owner_id': owner_id, 'attempt_id': attempt_id,
               'observed_at': sup.now(), 'status': 'confirmed'}
        if self.identity_hook:row = self.identity_hook(pid, row)
        self.identities.append(copy.deepcopy(row))
        return row
    def children(self, pid):
        return self.children_hook(pid) if self.children_hook else self.child_ids
    def rss(self, pid):return self.rss_hook(pid) if self.rss_hook else 1024
    def send_signal(self, expected, sig):
        self.signals.append((expected['pid'], sig))
        if sig in self.signal_fault:
            fault = self.signal_fault[sig]
            if isinstance(fault, ProcessLookupError):self.exit_at = self.clock+.01
            raise fault
        if self.stubborn:return
        if sig == signal.SIGKILL or not self.ignore_term:
            self.exit_at, self.exit_code = self.clock+.01, -sig


class RecordingBackend(sup.ProcessBackend):
    def spawn(self, *args, **kwargs):
        proc = super().spawn(*args, **kwargs)
        REAL['supervised_Popen'] += 1
        return proc
    def identity(self, *args):
        row = super().identity(*args)
        key = (row['pid'], row['create_time'])
        if key not in {(x['pid'], x['create_time']) for x in REAL['created_identities']}:
            REAL['created_identities'].append(copy.deepcopy(row))
        return row
    def send_signal(self, expected, sig):
        # Only fixture identities registered by this very execute_attempt are accepted.
        assert expected['owner_id'] == 'fixture-owner'
        REAL['signals'].append({'expected': copy.deepcopy(expected), 'signal': signal.Signals(sig).name})
        return super().send_signal(expected, sig)


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.root = OUT/self.id().split('.')[-1]
        self.root.mkdir(parents=True)
        self.records = []
        self.fallback = io.StringIO()
    def mock_run(self, backend=None, *, must_fail=False, **overrides):
        backend = backend or FakeBackend()
        def artifact():
            return {'status': 'completed' if backend.proc.returncode == 0 else 'absent',
                    'contract': 'unverified', 'refs': [{'fake_completion': 1}] if backend.proc.returncode == 0 else []}
        options = dict(directory=self.root/'attempts', command=['fake-python', 'fake-worker'], cwd=self.root,
                       session=session(), code={'commit': 'e'*40, 'tree': 'f'*40, 'dirty': False},
                       config_ref=None, stage='fixture', episode=0, owner_id='fixture-owner',
                       backend=backend, artifact=artifact, validate_artifact=lambda value: None,
                       monitor_interval_s=.02, wall_s=1., fallback=self.fallback)
        options.update(overrides)
        try:
            record = sup.execute_attempt(**options)
        except sup.SupervisionFailure as exc:
            record = exc.record
            if not must_fail:raise
        else:
            if must_fail:self.fail('expected non-success supervision')
        self.records.append(record)
        self.assertLess(backend.clock, 4.)
        for pid in {pid for pid, sig in backend.signals}:
            self.assertLessEqual(backend.signals.count((pid, signal.SIGTERM)), 1)
            self.assertLessEqual(backend.signals.count((pid, signal.SIGKILL)), 1)
        return record, backend
    def test_natural_exit_no_signals(self):
        record, backend = self.mock_run()
        self.assertEqual(record['supervisor_outcome'], 'passed')
        self.assertEqual(record['reap']['status'], 'reaped_by_popen')
        self.assertEqual(record['child_exit']['returncode'], 0)
        self.assertFalse(backend.signals)
    def test_terminal_postpublish_directory_fsync_failure_not_accepted(self):
        import stat
        real_fsync = os.fsync
        def save(path, value, exclusive=False):
            if value.get('phase') != 'terminal':
                return sup.atomic_json(path, value, exclusive)
            def failed(fd):
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    raise OSError(errno.EIO, 'fixture after terminal publication')
                return real_fsync(fd)
            with patch.object(sup.os, 'fsync', failed):
                return sup.atomic_json(path, value, exclusive)
        record, _ = self.mock_run(save=save, must_fail=True)
        self.assertEqual(record['supervisor_outcome'], 'failed')
        started = next((self.root/'attempts').glob('*.started.json'))
        self.assertEqual(sup.read_attempt(started)['supervisor_outcome'], 'unknown')
        self.assertRaises(ValueError, sup.assert_prior_attempts_closed, self.root/'attempts')
    def test_terminal_failure_after_successful_fsync_not_accepted(self):
        def save(path, value, exclusive=False):
            result = sup.atomic_json(path, value, exclusive)
            if value.get('phase') == 'terminal':
                raise OSError(errno.EIO, 'fixture after completed terminal fsync')
            return result
        self.mock_run(save=save, must_fail=True)
        started = next((self.root/'attempts').glob('*.started.json'))
        self.assertEqual(sup.read_attempt(started)['supervisor_outcome'], 'unknown')
        self.assertRaises(ValueError, sup.assert_prior_attempts_closed, self.root/'attempts')
    def test_successful_terminal_removes_commit_guard(self):
        self.mock_run()
        started = next((self.root/'attempts').glob('*.started.json'))
        self.assertEqual(sup.read_attempt(started)['supervisor_outcome'], 'passed')
        self.assertFalse(list((self.root/'attempts').glob('*.commit_pending')))
    def test_started_durable_before_spawn(self):
        backend = FakeBackend()
        def check():
            starts = list((self.root/'attempts').glob('*.started.json'))
            self.assertEqual(len(starts), 1)
            value = json.loads(starts[0].read_text())
            self.assertEqual(value['child_exit']['status'], 'not_spawned')
            self.assertFalse(list((self.root/'attempts').glob('*.json.tmp*')))
        backend.on_spawn = check
        self.mock_run(backend)
    def test_spawn_failure_terminal(self):
        backend = FakeBackend()
        backend.on_spawn = lambda: (_ for _ in ()).throw(OSError(errno.EAGAIN, 'spawn refused'))
        record, _ = self.mock_run(backend, must_fail=True)
        self.assertEqual(record['child_exit']['status'], 'not_spawned')
        self.assertEqual(record['errors']['primary_error']['errno'], errno.EAGAIN)
        self.assertTrue(list((self.root/'attempts').glob('*.started.json')))
    def test_leader_no_such_poll_none_then_popen_exit(self):
        backend = FakeBackend(exit_at=.015)
        backend.identity_hook = lambda *args: (_ for _ in ()).throw(psutil.NoSuchProcess(900001))
        record, backend = self.mock_run(backend)
        self.assertFalse(backend.signals)
        self.assertEqual(record['supervisor_outcome'], 'passed')
        self.assertEqual(record['telemetry']['events'][0]['source'], 'leader_identity')
        self.assertTrue(record['telemetry']['events'][0]['error'])
    def test_enumeration_no_such_recovered_by_wait(self):
        backend = FakeBackend(exit_at=.015)
        backend.children_hook = lambda *a: (_ for _ in ()).throw(psutil.NoSuchProcess(900001))
        record, backend = self.mock_run(backend)
        self.assertFalse(backend.signals)
        self.assertEqual(record['telemetry']['events'][0]['source'], 'children_enumeration')
    def test_single_child_disappears_only_child_sample_ends(self):
        backend = FakeBackend()
        backend.child_ids = [900002]
        backend.identity_hook = lambda pid,row: (_ for _ in ()).throw(psutil.NoSuchProcess(pid)) if pid == 900002 else row
        record, backend = self.mock_run(backend)
        self.assertEqual(record['supervisor_outcome'], 'passed')
        self.assertIn('child_identity', {x['source'] for x in record['telemetry']['events']})
        self.assertFalse(backend.signals)
    def test_telemetry_stays_unknown_no_unconditional_kill(self):
        backend = FakeBackend(exit_at=100.)
        backend.identity_hook = lambda *a: (_ for _ in ()).throw(psutil.NoSuchProcess(900001))
        record, backend = self.mock_run(backend, must_fail=True)
        self.assertFalse(backend.signals)
        self.assertIsNone(record['child_exit']['returncode'])
        self.assertEqual(record['cleanup']['status'], 'unverified')
        self.assertEqual(record['errors']['primary_error']['type'], 'TelemetryUnavailable')
    def test_access_denied_telemetry_is_not_gone(self):
        backend = FakeBackend(exit_at=100.)
        backend.identity_hook = lambda *a: (_ for _ in ()).throw(psutil.AccessDenied(900001))
        record, backend = self.mock_run(backend, must_fail=True)
        self.assertFalse(backend.signals)
        self.assertIn('denied', {x['status'] for x in record['telemetry']['events']})
    def test_transient_telemetry_failure_remains_visible_after_clean_reap(self):
        backend=FakeBackend(exit_at=.05)
        def identity(pid,row):
            if backend.clock>=.04:row['create_time']=101.
            return row
        backend.identity_hook=identity
        record,backend=self.mock_run(backend)
        self.assertEqual(record['supervisor_outcome'],'passed')
        self.assertEqual(record['telemetry']['status'],'gap')
        self.assertTrue(any(x['source']=='telemetry_retry' and 'mismatch' in x['error']['message'] for x in record['telemetry']['events']))
        self.assertFalse(backend.signals)
    def test_term_once_clean_stopped(self):
        record, backend = self.mock_run(FakeBackend(exit_at=100.), wall_s=.08, must_fail=True)
        self.assertEqual(record['supervisor_outcome'], 'stopped')
        self.assertEqual(backend.signals, [(900001, signal.SIGTERM)])
        self.assertEqual(record['reap']['status'], 'reaped_by_popen')
    def test_ignore_term_one_kill(self):
        record, backend = self.mock_run(FakeBackend(exit_at=100., ignore_term=True), wall_s=.08, must_fail=True)
        self.assertEqual(record['supervisor_outcome'], 'stopped')
        self.assertEqual(backend.signals, [(900001, signal.SIGTERM), (900001, signal.SIGKILL)])
    def test_kill_wait_timeout_unknown_not_passed(self):
        record, backend = self.mock_run(FakeBackend(exit_at=100., stubborn=True), wall_s=.08, must_fail=True)
        self.assertEqual(record['supervisor_outcome'], 'failed')
        self.assertIsNone(record['child_exit']['returncode'])
        self.assertEqual(record['reap']['status'], 'unverified')
    def test_eperm_term_preserves_primary_and_no_kill(self):
        backend = FakeBackend(exit_at=100.)
        backend.signal_fault[signal.SIGTERM] = PermissionError(errno.EPERM, 'mock EPERM')
        record, backend = self.mock_run(backend, wall_s=.08, must_fail=True)
        self.assertEqual(backend.signals, [(900001, signal.SIGTERM)])
        self.assertEqual(record['errors']['primary_error']['type'], 'StopRequested')
        self.assertEqual(record['errors']['cleanup_errors'][0]['errno'], errno.EPERM)
        self.assertEqual(record['supervisor_outcome'], 'failed')
    def test_accessdenied_kill_preserves_failure(self):
        backend = FakeBackend(exit_at=100., ignore_term=True)
        backend.signal_fault[signal.SIGKILL] = psutil.AccessDenied(900001)
        record, backend = self.mock_run(backend, wall_s=.08, must_fail=True)
        self.assertEqual(len(backend.signals), 2)
        self.assertEqual(record['errors']['cleanup_errors'][0]['type'], 'AccessDenied')
        self.assertEqual(record['supervisor_outcome'], 'failed')
    def test_psutil_accessdenied_retains_underlying_eperm(self):
        try:
            try:raise PermissionError(errno.EPERM,'OS denied signal')
            except PermissionError as original:raise psutil.AccessDenied(900001) from original
        except psutil.AccessDenied as exc:
            value=sup.error(exc,'signal')
        self.assertEqual(value['type'],'AccessDenied')
        self.assertEqual(value['errno'],errno.EPERM)
        self.assertIn('OS denied signal',value['traceback'])
    def test_nondefault_sigchld_denied_before_spawn(self):
        original=signal.getsignal(signal.SIGCHLD)
        signal.signal(signal.SIGCHLD,signal.SIG_IGN)
        try:
            record,backend=self.mock_run(must_fail=True)
            self.assertEqual(backend.spawn_count,0)
            self.assertIn('SIGCHLD',record['errors']['primary_error']['message'])
        finally:signal.signal(signal.SIGCHLD,original)
    def test_esrch_does_not_manufacture_exit_code(self):
        backend = FakeBackend(exit_at=100.)
        backend.signal_fault[signal.SIGTERM] = ProcessLookupError(errno.ESRCH, 'mock exit race')
        record, backend = self.mock_run(backend, wall_s=.08, must_fail=True)
        self.assertEqual(record['cleanup']['signals'][0]['result'], 'ESRCH')
        self.assertEqual(record['reap']['status'], 'reaped_by_popen')
        self.assertTrue(record['reap']['observations'])
    def test_primary_cleanup_evidence_all_survive(self):
        backend = FakeBackend(exit_at=100.)
        backend.signal_fault[signal.SIGTERM] = PermissionError(errno.EPERM, 'cleanup')
        def primary(*a):raise RuntimeError('original control failure')
        def evidence():raise OSError('evidence failure')
        record, backend = self.mock_run(backend, resource_check=primary, evidence=evidence, must_fail=True)
        self.assertIn('original control failure', record['errors']['primary_error']['message'])
        self.assertTrue(record['errors']['cleanup_errors'])
        self.assertEqual(record['errors']['evidence_errors'][0]['message'], 'evidence failure')
    def test_nonzero_even_with_completed_artifact(self):
        backend = FakeBackend();backend.exit_code = 7
        def artifact():return {'status':'completed' if backend.proc.returncode is not None else 'absent','refs':[1] if backend.proc.returncode is not None else [],'contract':'unverified'}
        record, backend = self.mock_run(backend, artifact=artifact, must_fail=True)
        self.assertEqual(record['child_exit']['returncode'], 7)
        self.assertEqual(record['supervisor_outcome'], 'failed')
    def test_identity_changes_refuse_each_signal(self):
        for key, changed in [('create_time', 101.), ('create_time', None), ('uid', os.getuid()+1),
                             ('pgid', 910001), ('sid', 910001), ('ppid', 1), ('status', 'zombie')]:
            with self.subTest(key=key, changed=changed):
                root = self.root/(key+str(changed));root.mkdir()
                original_root, self.root = self.root, root
                backend = FakeBackend(exit_at=100.)
                def identity(pid,row):
                    if backend.clock >= .08:row[key] = changed
                    return row
                backend.identity_hook = identity
                try:
                    record, _ = self.mock_run(backend, wall_s=.08, must_fail=True)
                    self.assertFalse(backend.signals)
                    self.assertNotEqual(record['supervisor_outcome'], 'passed')
                    if changed == 'zombie':self.assertEqual(record['reap']['status'], 'unverified')
                finally:self.root = original_root
    def test_fresh_process_backend_rechecks_before_signal(self):
        expected = {'pid':900001,'create_time':100.,'ppid':os.getpid(),'uid':os.getuid(),'pgid':900001,'sid':900001,'owner_id':'fixture-owner','attempt_id':'id','status':'confirmed'}
        changed = {**expected,'create_time':101.}
        process = SimpleNamespace(send_signal=lambda *a:self.fail('unsafe signal'))
        backend = sup.ProcessBackend()
        with patch.object(sup.psutil,'Process',return_value=process) as factory,patch.object(backend,'_identity',return_value=changed):
            self.assertRaises(sup.IdentityRefused,backend.send_signal,expected,signal.SIGTERM)
            factory.assert_called_once_with(900001)
    def test_wait_error_keeps_reap_unverified(self):
        backend = FakeBackend(exit_at=.02);backend.wait_error=OSError(errno.ECHILD,'foreign reaper or wait fault')
        record, backend = self.mock_run(backend, must_fail=True)
        self.assertEqual(record['supervisor_outcome'],'failed')
        self.assertEqual(record['reap']['status'],'unverified')
        self.assertTrue(record['errors']['cleanup_errors'])
        self.assertFalse(backend.signals)
        self.assertEqual(len(record['errors']['cleanup_errors']),1)
    def test_postflight_failure_and_evidence_failure_both_recorded(self):
        def postflight():raise ValueError('source changed')
        def evidence():raise OSError('index write failed')
        record,backend=self.mock_run(postflight=postflight,evidence=evidence,must_fail=True)
        self.assertEqual([x['operation'] for x in record['errors']['evidence_errors']],['postflight','top_level_evidence'])
        self.assertFalse(backend.signals)
    def test_stop_latch_failure_keeps_stop_and_cleanup(self):
        def latch(exc):raise OSError('latch fault')
        record,backend=self.mock_run(FakeBackend(exit_at=100.),wall_s=.08,on_stop=latch,must_fail=True)
        self.assertEqual(record['errors']['primary_error']['type'],'StopRequested')
        self.assertEqual(record['errors']['evidence_errors'][0]['operation'],'resource_stop_latch')
        self.assertEqual(record['supervisor_outcome'],'failed')
    def test_terminal_write_failure_stderr_and_unknown_start(self):
        def save(path,value,exclusive=False):
            if str(path).endswith('.started.json'):return sup.atomic_json(path,value,exclusive)
            raise OSError(errno.ENOSPC,'terminal storage fault')
        record,backend=self.mock_run(save=save,must_fail=True)
        self.assertEqual(record['errors']['receipt_errors'][0]['errno'],errno.ENOSPC)
        self.assertIn('terminal storage fault',self.fallback.getvalue())
        started=next((self.root/'attempts').glob('*.started.json'))
        self.assertEqual(sup.read_attempt(started)['supervisor_outcome'],'unknown')
        self.assertRaises(ValueError,sup.assert_prior_attempts_closed,self.root/'attempts')
    def test_started_write_failure_spawns_nothing(self):
        def save(path,value,exclusive=False):
            if str(path).endswith('.started.json'):raise OSError(errno.ENOSPC,'no started storage')
            return sup.atomic_json(path,value,exclusive)
        record,backend=self.mock_run(save=save,must_fail=True)
        self.assertEqual(backend.spawn_count,0)
        self.assertEqual(record['child_exit']['status'],'not_spawned')
    def test_reserve_failure_spawns_nothing(self):
        original=Path.open
        def opened(path,*a,**k):
            if path.suffix=='.reserve':raise OSError(errno.ENOSPC,'no reserve')
            return original(path,*a,**k)
        with patch.object(Path,'open',opened):record,backend=self.mock_run(must_fail=True)
        self.assertEqual(backend.spawn_count,0)
        self.assertEqual(record['errors']['primary_error']['errno'],errno.ENOSPC)
    def test_handler_restore_failure_recorded(self):
        original=sup.signal.signal;calls={}
        def changed(sig,handler):
            calls[sig]=calls.get(sig,0)+1
            if calls[sig]>1:raise OSError('handler restore fault')
            return original(sig,handler)
        saved={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            with patch.object(sup.signal,'signal',changed):record,backend=self.mock_run(must_fail=True)
            self.assertEqual(len(record['errors']['handler_errors']),2)
            self.assertFalse(backend.signals)
        finally:
            for sig,handler in saved.items():original(sig,handler)
    def test_invalid_and_expired_policies_no_spawn(self):
        values=[]
        for field,value in [('term_wait_s',float('nan')),('kill_wait_s',-1),('cleanup_total_s',.1),('telemetry_retry_count',1.5)]:
            s=session();s['cleanup'][field]=value;values.append(s)
        s=session();s['hard_stop']='2000-01-01T00:00:00+08:00';values.append(s)
        for i,s in enumerate(values):
            old,self.root=self.root,self.root/str(i);self.root.mkdir()
            try:
                record,backend=self.mock_run(session=s,must_fail=True)
                self.assertEqual(backend.spawn_count,0)
            finally:self.root=old
    def test_artifact_contract_error_blocks_pass(self):
        def validate(value):raise ValueError('invalid worker hash/coverage')
        record,backend=self.mock_run(validate_artifact=validate,must_fail=True)
        self.assertIn('invalid worker hash',record['errors']['primary_error']['message'])
    def test_stale_completion_cannot_pass_new_attempt(self):
        record,backend=self.mock_run(artifact=lambda:{'status':'completed','refs':[1]},must_fail=True)
        self.assertIn('did not change',record['errors']['primary_error']['message'])
    def test_control_owner_evidence_and_primary_errors_survive(self):
        class BadOwner:
            def __enter__(self):
                self.fd=(self_root/'owner.lock').open('a+');return self
            def __exit__(self,*a):raise OSError('Owner end write fault')
        self_root=self.root;owner=BadOwner()
        def evidence():raise OSError('controller evidence fault')
        with self.assertRaises(sup.SupervisionFailure) as caught:
            with sup.control_scope(owner,self.root/'control',evidence=evidence):
                raise RuntimeError('controller primary')
        value=caught.exception.record
        self.assertEqual(value['errors']['primary_error']['message'],'controller primary')
        self.assertEqual(value['errors']['owner_errors'][0]['message'],'Owner end write fault')
        self.assertEqual(value['errors']['evidence_errors'][0]['message'],'controller evidence fault')
        self.assertTrue(owner.fd.closed)
        self.records.append(value)
    def test_control_owner_enter_failure_releases_fd(self):
        class BadOwner:
            def __enter__(self):
                self.fd=(self_root/'owner.lock').open('a+');raise OSError('Owner enter fault')
        self_root=self.root;owner=BadOwner()
        with self.assertRaises(sup.SupervisionFailure) as caught:
            with sup.control_scope(owner,self.root/'control'):self.fail('must not enter')
        self.assertTrue(owner.fd.closed)
        self.assertEqual(caught.exception.record['errors']['primary_error']['message'],'Owner enter fault')
    def test_owner_refusal_does_not_run_writing_evidence(self):
        class RefusedOwner:
            def __enter__(self):raise ValueError('active owner')
        with self.assertRaises(sup.SupervisionFailure):
            with sup.control_scope(RefusedOwner(),self.root/'control',evidence=lambda:self.fail('no ownership for evidence write')):
                self.fail('must not enter')
    def test_writing_evidence_runs_while_actual_owner_lock_held(self):
        import fcntl
        import r11_state
        owner=r11_state.Owner(self.root/'owner','fixture',None)
        calls=[]
        def evidence():
            self.assertFalse(owner.fd.closed)
            with (self.root/'owner/owner.lock').open('a+') as competing:
                self.assertRaises(BlockingIOError,fcntl.flock,competing,fcntl.LOCK_EX|fcntl.LOCK_NB)
            calls.append('under_owner_lock')
        with sup.control_scope(owner,self.root/'control',evidence=evidence):pass
        self.assertEqual(calls,['under_owner_lock'])
        self.assertTrue(owner.fd.closed)
    def test_actual_owner_end_write_failure_preserves_primary_and_unlocks(self):
        import fcntl
        import r11_state
        owner=r11_state.Owner(self.root/'owner','fixture',None)
        original=r11_state.atomic_json
        def save(path,*args,**kwargs):
            if str(path).endswith('_end.json'):raise OSError('actual Owner end fault')
            return original(path,*args,**kwargs)
        with patch.object(r11_state,'atomic_json',save):
            with self.assertRaises(sup.SupervisionFailure) as caught:
                with sup.control_scope(owner,self.root/'control'):
                    raise RuntimeError('original Owner body fault')
        self.assertTrue(owner.fd.closed)
        record=caught.exception.record
        self.assertEqual(record['errors']['primary_error']['message'],'original Owner body fault')
        self.assertEqual(record['errors']['owner_errors'][0]['message'],'actual Owner end fault')
        with (self.root/'owner/owner.lock').open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(stream,fcntl.LOCK_UN)
        self.records.append(record)
    def test_controller_start_is_not_mistaken_for_worker_attempt(self):
        class Owner:
            def __enter__(self):return self
            def __exit__(self,*args):pass
        with sup.control_scope(Owner(),self.root/'attempts/controller_fixture'):
            record,backend=self.mock_run()
            self.assertEqual(record['supervisor_outcome'],'passed')
    def test_control_terminal_write_failure_nonzero_fallback(self):
        class Owner:
            def __enter__(self):return self
            def __exit__(self,*a):pass
        def save(path,value,exclusive=False):
            if str(path).endswith('.started.json'):return sup.atomic_json(path,value,exclusive)
            raise OSError('controller terminal fault')
        with self.assertRaises(sup.SupervisionFailure):
            with sup.control_scope(Owner(),self.root/'control',save=save,fallback=self.fallback):pass
        self.assertIn('controller terminal fault',self.fallback.getvalue())
    def test_adapter_missing_session_denies_before_ledger(self):
        import r11
        cfg={'stage_limits':{'A':{'wall_s':1,'rss_bytes':10000}},'monitor_interval_s':.02}
        man={'code':{'commit':'e'*40,'tree':'f'*40},'kit_source_hashes':{}}
        with patch.object(r11,'Ledger',side_effect=AssertionError('must not open production ledger')):
            with self.assertRaises(sup.SupervisionFailure) as caught:
                r11.supervise(self.root,cfg,man,'A',0,owner_id='fixture-owner')
        self.assertEqual(caught.exception.record['child_exit']['status'],'not_spawned')
    def real_run(self,mode,*,must_fail=False,wall_s=1.,resource_check=None):
        fixture=self.root/'fixture';fixture.mkdir()
        command=[sys.executable,'-B',str(SOURCE/'tests/supervision_dummy.py'),mode,str(fixture)]
        def artifact():
            path=fixture/'completed.json'
            return {'status':'completed' if path.exists() else 'absent','contract':'unverified','refs':[sup.file_ref(path)] if path.exists() else []}
        start=time.monotonic()
        try:
            record=sup.execute_attempt(directory=self.root/'attempts',command=command,cwd=self.root,
                session=session(),code={'commit':HEAD,'tree':TREE,'dirty':False},config_ref=None,
                stage='dummy',episode=None,owner_id='fixture-owner',backend=RecordingBackend(),
                artifact=artifact,validate_artifact=lambda value:json.loads((fixture/'completed.json').read_text()),
                wall_s=wall_s,monitor_interval_s=.02,resource_check=resource_check)
        except sup.SupervisionFailure as exc:
            if not must_fail:raise
            record=exc.record
        else:
            if must_fail:self.fail('real fixture should not pass')
        self.assertLess(time.monotonic()-start,3.)
        self.records.append(record)
        return record,fixture
    def test_real_natural_exit(self):
        record,fixture=self.real_run('natural')
        self.assertEqual(record['child_exit']['returncode'],0)
        self.assertFalse(record['cleanup']['signals'])
    def test_real_completed_then_late_exit(self):
        record,fixture=self.real_run('late')
        self.assertEqual(record['supervisor_outcome'],'passed')
        self.assertGreater(record['elapsed_s'],.1)
        self.assertFalse(record['cleanup']['signals'])
    def test_real_term_once(self):
        record,fixture=self.real_run('term',wall_s=.25,must_fail=True)
        self.assertEqual(record['supervisor_outcome'],'stopped')
        self.assertEqual([x['signal'] for x in record['cleanup']['signals']],['SIGTERM'])
    def test_real_ignores_term_then_kill(self):
        record,fixture=self.real_run('ignore',wall_s=.25,must_fail=True)
        self.assertEqual(record['supervisor_outcome'],'stopped')
        self.assertEqual(record['child_exit']['returncode'],-signal.SIGKILL)
        self.assertEqual([x['signal'] for x in record['cleanup']['signals']],['SIGTERM','SIGKILL'])
    def test_real_parent_child_grandchild_registered(self):
        record,fixture=self.real_run('tree',wall_s=.4,must_fail=True)
        self.assertGreaterEqual(len(record['registered_targets']),3)
        self.assertEqual(record['supervisor_outcome'],'stopped')
        self.assertEqual(record['group_observation']['status'],'registered_targets_gone')
        self.assertEqual(len(record['cleanup']['signals']),3)
    def test_real_leader_exits_before_child(self):
        fixture=self.root/'fixture'
        def acknowledge(*a):
            if (fixture/'orphan_ready').exists():(fixture/'registered_ack').write_text('registered')
        record,fixture=self.real_run('leader-first',resource_check=acknowledge)
        self.assertGreaterEqual(len(record['registered_targets']),2)
        self.assertEqual(record['supervisor_outcome'],'passed')
        self.assertFalse(record['cleanup']['signals'])
        self.assertEqual(record['group_observation']['status'],'registered_targets_gone')
    def test_real_nonzero_completed_still_fails(self):
        record,fixture=self.real_run('nonzero',must_fail=True)
        self.assertEqual(record['child_exit']['returncode'],7)
        self.assertEqual(record['supervisor_outcome'],'failed')
    def test_real_controller_crash_started_remains_unknown(self):
        fixture=self.root/'crash';fixture.mkdir()
        driver=subprocess.Popen([sys.executable,'-B',str(SOURCE/'tests/supervision_dummy.py'),'crash-controller',str(fixture)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        REAL['fixture_driver_Popen']+=1
        stdout,stderr=driver.communicate(timeout=4.)
        (self.root/'driver.stdout.log').write_bytes(stdout);(self.root/'driver.stderr.log').write_bytes(stderr)
        self.assertEqual(driver.returncode,77)
        started=next((fixture/'attempts').glob('*.started.json'))
        self.assertEqual(sup.read_attempt(started)['supervisor_outcome'],'unknown')
        self.assertRaises(ValueError,sup.assert_prior_attempts_closed,fixture/'attempts')
        self.assertFalse(started.with_name(started.name.replace('.started.json','.json')).exists())
        time.sleep(.25)  # The fixture child has a finite .04s lifetime; no cleanup signals.


class EvidenceResult(unittest.TextTestResult):
    def startTest(self,test):
        self.started=time.monotonic();super().startTest(test)
    def stopTest(self,test):
        failed=any(case is test for case,_ in self.failures+self.errors)
        RESULTS.append({'name':test.id().split('.')[-1],'status':'failed' if failed else 'passed',
                        'kind':'real_OS' if 'test_real_' in test.id() else 'fault_stub_or_contract',
                        'elapsed_s':time.monotonic()-self.started,
                        'record_count':len(getattr(test,'records',[]))})
        super().stopTest(test)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
    HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    TREE=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=SOURCE,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip()
    result=unittest.TextTestRunner(verbosity=2,resultclass=EvidenceResult).run(unittest.defaultTestLoader.loadTestsFromTestCase(SupervisionTests))
    payload={'schema':'T012.04_supervision_checks_v1','created_at':sup.now(),
             'status':'passed' if result.wasSuccessful() else 'failed','commit':HEAD,'tree':TREE,'dirty_at_start':dirty,
             'python':sys.executable,'python_version':sys.version,'psutil_version':psutil.__version__,
             'cases':RESULTS,'real_OS':REAL,'forbidden_operations':AUDIT['forbidden'],
             'real_model_loads':0,'real_model_forwards':0,'real_video_decodes':0,
             'failures':[{'name':case.id(),'traceback':tb} for case,tb in result.failures+result.errors],
             'source_refs':[sup.file_ref(path) for path in (SOURCE/'r11.py',SOURCE/'r12_supervision.py',Path(__file__),SOURCE/'tests/supervision_dummy.py')]}
    sup.atomic_json(OUT/'results.json',payload,True)
    print(json.dumps({'status':payload['status'],'cases':len(RESULTS),'result':str(OUT/'results.json')},indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
