"""Bounded, identity-checked supervision. No model, video or numerical imports.

The caller owns session admission, budget and the worker artifact contract.
Only this module's Popen may reap its direct child. Descendants are observed,
never waited with waitpid/psutil.wait. Group signalling is deliberately absent.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import datetime
import errno
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
import uuid

import psutil


TZ = datetime.timezone(datetime.timedelta(hours=8))
ERROR_LISTS = ('cleanup_errors', 'receipt_errors', 'evidence_errors',
               'owner_errors', 'handler_errors')
IDENTITY_KEYS = ('pid', 'create_time', 'ppid', 'uid', 'pgid', 'sid',
                 'owner_id', 'attempt_id')


def now():
    return datetime.datetime.now(TZ).isoformat()


def error(exc, operation, target=None):
    cause, seen, number = exc, set(), None
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        number = getattr(cause, 'errno', None)
        if number is not None:
            break
        cause = cause.__cause__ or cause.__context__
    return {'type': type(exc).__name__, 'message': str(exc),
            'errno': number,
            'traceback': ''.join(traceback.format_exception(exc)),
            'operation': operation, 'observed_at': now(),
            'target_identity': target}


def errors():
    return {'primary_error': None, **{key: [] for key in ERROR_LISTS}}


def file_ref(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def atomic_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


class StopRequested(RuntimeError):
    """Explicit wall/resource/user stop; successful cleanup is still non-success."""


class SupervisionFailure(RuntimeError):
    def __init__(self, record, path):
        self.record, self.path = record, Path(path)
        super().__init__('supervision ' + record['supervisor_outcome'] + ': ' + str(path))


class TelemetryUnavailable(RuntimeError):
    pass


class IdentityRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class Policy:
    exit_grace_s: float
    term_wait_s: float
    kill_wait_s: float
    cleanup_total_s: float
    telemetry_retry_count: int
    telemetry_retry_total_s: float

    @classmethod
    def from_session(cls, session):
        policy = cls(**{key: session['cleanup'][key] for key in cls.__dataclass_fields__})
        for key, value in vars(policy).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError('finite nonnegative cleanup parameter required: ' + key)
        if type(policy.telemetry_retry_count) is not int:
            raise ValueError('telemetry_retry_count must be an integer')
        if policy.cleanup_total_s < policy.exit_grace_s + policy.term_wait_s + policy.kill_wait_s + policy.telemetry_retry_total_s:
            raise ValueError('cleanup_total_s must cover grace, TERM, KILL and telemetry waits')
        return policy


def same_identity(expected, actual):
    return (actual['status'] == 'confirmed' and
            all(expected.get(key) is not None and actual.get(key) == expected[key]
                for key in IDENTITY_KEYS))


class ProcessBackend:
    """Every query constructs a fresh Process; no cached identity authorizes a signal."""
    monotonic = staticmethod(time.monotonic)
    wall_time = staticmethod(time.time)
    sleep = staticmethod(time.sleep)
    spawn = staticmethod(subprocess.Popen)

    def _identity(self, process, owner_id, attempt_id):
        identity = {'pid': process.pid, 'create_time': process.create_time(),
                    'ppid': process.ppid(), 'uid': process.uids().real,
                    'pgid': os.getpgid(process.pid), 'sid': os.getsid(process.pid),
                    'owner_id': owner_id, 'attempt_id': attempt_id,
                    'observed_at': now(), 'status': 'confirmed'}
        status = process.status()
        if status == psutil.STATUS_ZOMBIE:
            identity['status'] = 'zombie'
        elif (not isinstance(identity['create_time'], (float, int)) or
              not math.isfinite(identity['create_time']) or identity['create_time'] <= 0 or
              any(identity[key] is None for key in IDENTITY_KEYS)):
            identity['status'] = 'unknown'
        return identity

    def identity(self, pid, owner_id, attempt_id):
        return self._identity(psutil.Process(pid), owner_id, attempt_id)

    def children(self, pid):
        return [child.pid for child in psutil.Process(pid).children(recursive=True)]

    def rss(self, pid):
        return psutil.Process(pid).memory_info().rss

    def send_signal(self, expected, sig):
        fresh = psutil.Process(expected['pid'])
        actual = self._identity(fresh, expected['owner_id'], expected['attempt_id'])
        if not same_identity(expected, actual):
            raise IdentityRefused('identity changed between verification and send_signal')
        fresh.send_signal(sig)


def read_attempt(started_path):
    """A durable start without terminal is unknown; never synthesize an exit status."""
    started_path = Path(started_path)
    terminal = started_path.with_name(started_path.name.replace('.started.json', '.json'))
    value = json.loads((terminal if terminal.exists() else started_path).read_text())
    if not terminal.exists():
        value['supervisor_outcome'] = 'unknown'
        if 'child_exit' in value:
            value['child_exit']['status'] = 'unknown'
            value['reap']['status'] = 'unverified'
    return value


def assert_prior_attempts_closed(directory, exclude=None):
    for started in Path(directory).glob('*.started.json'):
        if json.loads(started.read_text()).get('schema') != 'r12_supervision_attempt_v1':
            continue  # The enclosing controller/Owner is intentionally still active.
        value = read_attempt(started)
        if value['attempt_id'] != exclude and value['supervisor_outcome'] != 'passed':
            raise ValueError('prior supervision requires independent resolution: ' + str(started))


class _Attempt:
    def __init__(self, directory, command, cwd, session, code, config_ref,
                 stage, episode, owner_id, parent_attempt, parent_refs,
                 backend, save, fallback):
        self.backend, self.save, self.fallback = backend, save, fallback
        self.started = backend.monotonic()
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        aid = uuid.uuid4().hex
        self.prefix = self.directory / f'{stage}_{episode}_{aid}'
        self.terminal = self.prefix.with_suffix('.json')
        self.start_path = self.prefix.with_suffix('.started.json')
        self.reserve = self.prefix.with_suffix('.reserve')
        self.record = {'schema': 'r12_supervision_attempt_v1', 'phase': 'started',
                       'session_id': session.get('session_id'), 'attempt_id': aid,
                       'owner_id': owner_id, 'parent_attempt': parent_attempt,
                       'parent_refs': parent_refs, 'created_at': now(), 'code': code,
                       'config_ref': config_ref, 'session_ref': session.get('session_ref'),
                       'session': session, 'stage': stage, 'episode': episode,
                       'command': command, 'cwd': str(Path(cwd).resolve()),
                       'python': command[0] if command else None,
                       'worker_artifact': {'status': 'absent', 'contract': 'unverified', 'refs': []},
                       'child_exit': {'status': 'not_spawned', 'returncode': None, 'observations': []},
                       'reap': {'status': 'pending', 'observations': []},
                       'cleanup': {'status': 'not_needed', 'signals': [], 'observations': []},
                       'group_observation': {'status': 'unknown', 'scope': 'registered_targets_only'},
                       'telemetry': {'status': 'healthy', 'events': [], 'sample_count': 0},
                       'registered_targets': [], 'supervisor_outcome': 'unknown',
                       'errors': errors(), 'sampled_peak_tree_rss_bytes': 0,
                       'direct_child_reaper': 'this_Popen_only', 'group_signals_enabled': False}
        self.registry, self.proc, self.policy = {}, None, None
        self.primary, self.policy_stop, self.no_active_cleanup = None, False, False
        self.denied, self.handlers, self.signal_attempts = set(), [], set()
        self.cleanup_deadline = None
        self.wait_failed = False

    def event(self, source, status, pid=None, exc=None):
        events = self.record['telemetry']['events']
        if len(events) < 256:
            events.append({'source': source, 'status': status, 'pid': pid,
                           'observed_at': now(), 'error': error(exc, source) if exc else None})
        self.record['telemetry']['status'] = 'gap'

    def observe(self, pid, source):
        problem = None
        try:
            return self.backend.identity(pid, self.record['owner_id'], self.record['attempt_id'])
        except psutil.ZombieProcess as exc:
            status, problem = 'zombie', exc
        except (psutil.NoSuchProcess, ProcessLookupError) as exc:
            status, problem = 'gone', exc
        except (psutil.AccessDenied, PermissionError) as exc:
            status, problem = 'denied', exc
        except Exception as exc:
            status, problem = 'unknown', exc
        self.event(source, status, pid, problem)
        return {'pid': pid, 'status': status, 'observed_at': now()}

    def poll(self):
        rc = self.proc.poll()
        if rc is not None:
            self.record['child_exit'].update(status='exited', returncode=rc)
            self.record['child_exit']['observations'].append({'source': 'Popen.poll', 'returncode': rc, 'observed_at': now()})
        return rc

    def wait(self, seconds, source):
        if self.wait_failed:
            return False
        seconds = max(0, seconds)
        try:
            rc = self.proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            self.record['reap']['observations'].append({'source': source, 'status': 'timeout', 'timeout_s': seconds})
            return False
        except BaseException as exc:
            self.record['errors']['cleanup_errors'].append(error(exc, source))
            self.record['reap']['status'] = 'unverified'
            self.wait_failed = True
            return False
        self.record['child_exit'].update(status='exited', returncode=rc)
        self.record['reap']['status'] = 'reaped_by_popen'
        self.record['reap']['observations'].append({'source': source, 'status': 'returned', 'returncode': rc, 'timeout_s': seconds, 'observed_at': now()})
        return True

    def sample(self):
        pid = self.proc.pid
        leader = self.observe(pid, 'leader_identity')
        if leader['status'] != 'confirmed':
            raise TelemetryUnavailable('leader identity ' + leader['status'])
        if pid not in self.registry:
            if (leader['ppid'] != os.getpid() or leader['uid'] != os.getuid() or
                    leader['pgid'] != pid or leader['sid'] != pid):
                raise TelemetryUnavailable('spawned leader ownership/session mismatch')
            self.registry[pid] = leader
        elif not same_identity(self.registry[pid], leader):
            raise TelemetryUnavailable('leader identity mismatch')
        try:
            descendants = self.backend.children(pid)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied, ProcessLookupError, PermissionError) as exc:
            self.event('children_enumeration', type(exc).__name__, pid, exc)
            raise TelemetryUnavailable('children enumeration unavailable') from exc
        fresh = {}
        for child_pid in descendants:
            identity = self.observe(child_pid, 'child_identity')
            if identity['status'] in ('gone', 'zombie'):
                continue
            if identity['status'] != 'confirmed':
                raise TelemetryUnavailable('child identity ' + identity['status'])
            fresh[child_pid] = identity
        if len(set(self.registry) | fresh.keys()) > 128:
            raise TelemetryUnavailable('registration bound exceeded')
        # Register only a complete observed lineage; enumeration order is not trusted.
        pending = dict(fresh)
        while pending:
            progress = False
            for child_pid, identity in list(pending.items()):
                if identity['ppid'] not in self.registry:
                    continue
                parent = self.registry[identity['ppid']]
                actual_parent = leader if identity['ppid'] == pid else fresh.get(identity['ppid'])
                if actual_parent is None or not same_identity(parent, actual_parent):
                    raise TelemetryUnavailable('child parent identity unavailable')
                if any(identity[key] != leader[key] for key in ('uid', 'pgid', 'sid')):
                    raise TelemetryUnavailable('child ownership/session mismatch')
                if child_pid in self.registry and not same_identity(self.registry[child_pid], identity):
                    raise TelemetryUnavailable('registered child identity mismatch')
                self.registry.setdefault(child_pid, identity)
                del pending[child_pid]
                progress = True
            if not progress:
                raise TelemetryUnavailable('incomplete child lineage')
        resident = 0
        for target_pid in [pid, *fresh]:
            try:
                resident += self.backend.rss(target_pid)
            except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError) as exc:
                self.event('leader_rss' if target_pid == pid else 'child_rss', 'gone', target_pid, exc)
                if target_pid == pid:
                    raise TelemetryUnavailable('leader RSS unavailable') from exc
            except (psutil.AccessDenied, PermissionError) as exc:
                self.event('leader_rss' if target_pid == pid else 'child_rss', 'denied', target_pid, exc)
                raise TelemetryUnavailable('RSS denied') from exc
        self.record['telemetry']['sample_count'] += 1
        self.record['sampled_peak_tree_rss_bytes'] = max(resident, self.record['sampled_peak_tree_rss_bytes'])
        return resident

    def live_targets(self):
        live = {}
        for pid, expected in self.registry.items():
            if pid == self.proc.pid and self.record['reap']['status'] == 'reaped_by_popen':
                continue
            if pid == self.proc.pid and self.record['child_exit']['status'] == 'exited':
                live[pid] = {'pid': pid, 'status': 'reap_unverified'}
                continue
            actual = self.observe(pid, 'cleanup_observation')
            if actual['status'] != 'gone':
                if actual['status'] == 'confirmed' and not same_identity(expected, actual):
                    actual['status'] = 'mismatch'
                live[pid] = actual
        if self.proc and self.record['reap']['status'] != 'reaped_by_popen' and self.proc.pid not in live:
            # An absent process is not proof that Popen has reaped it.
            live[self.proc.pid] = {'pid': self.proc.pid, 'status': 'reap_unverified'}
        return live

    def passive_wait(self, seconds):
        end = min(self.cleanup_deadline, self.backend.monotonic() + seconds)
        while True:
            if self.record['reap']['status'] != 'reaped_by_popen':
                self.wait(min(.02, max(0, end - self.backend.monotonic())), 'Popen.cleanup_wait')
            if not self.live_targets() or self.backend.monotonic() >= end:
                return
            self.backend.sleep(min(.02, max(0, end - self.backend.monotonic())))

    def send(self, sig):
        # Descendants first; a cooperating leader can reap them before it exits.
        targets = [pid for pid in self.registry if pid != self.proc.pid]
        targets.sort(key=lambda pid: self.depth(pid), reverse=True)
        targets += [self.proc.pid] if self.proc.pid in self.registry else []
        for pid in targets:
            if self.backend.monotonic() >= self.cleanup_deadline or pid in self.denied or (pid, sig) in self.signal_attempts:
                continue
            if pid == self.proc.pid and self.record['reap']['status'] == 'reaped_by_popen':
                continue
            if pid == self.proc.pid and self.record['child_exit']['status'] == 'exited':
                continue
            expected = self.registry[pid]
            actual = self.observe(pid, 'pre_' + signal.Signals(sig).name)
            if actual['status'] in ('gone', 'zombie'):
                self.record['cleanup']['observations'].append(actual)
                continue
            if not same_identity(expected, actual):
                self.record['errors']['cleanup_errors'].append(error(IdentityRefused('signal refused: ' + actual['status']), 'identity_before_signal', actual))
                self.denied.add(pid)
                continue
            event = {'pid': pid, 'signal': signal.Signals(sig).name,
                     'identity': actual, 'observed_at': now(), 'result': 'attempted'}
            self.record['cleanup']['signals'].append(event)
            self.signal_attempts.add((pid, sig))
            try:
                self.backend.send_signal(expected, sig)
                event['result'] = 'sent'
            except (psutil.NoSuchProcess, ProcessLookupError) as exc:
                event.update(result='ESRCH', error=error(exc, 'send_signal', actual))
            except BaseException as exc:
                event.update(result='denied' if isinstance(exc, (PermissionError, psutil.AccessDenied)) else 'failed', error=error(exc, 'send_signal', actual))
                self.record['errors']['cleanup_errors'].append(event['error'])
                self.denied.add(pid)

    def depth(self, pid):
        seen = set()
        while pid in self.registry and pid not in seen:
            seen.add(pid)
            pid = self.registry[pid]['ppid']
        return len(seen)

    def cleanup(self):
        self.cleanup_deadline = min(self.hard_deadline, self.backend.monotonic() + self.policy.cleanup_total_s)
        self.record['cleanup']['monotonic_deadline'] = self.cleanup_deadline
        if not self.proc:
            return
        self.wait(0, 'Popen.finalize_wait')
        if self.primary is None or self.record['worker_artifact']['status'] == 'completed' or self.no_active_cleanup:
            self.passive_wait(self.policy.exit_grace_s)
        pending = self.live_targets()
        if pending and self.primary is None:
            self.set_primary(StopRequested('registered targets outlived exit grace'), 'exit_grace', ())
        if pending and not self.no_active_cleanup:
            self.send(signal.SIGTERM)
            self.passive_wait(self.policy.term_wait_s)
            if self.live_targets():
                self.send(signal.SIGKILL)
                self.passive_wait(self.policy.kill_wait_s)
        elif pending:
            self.passive_wait(self.policy.term_wait_s + self.policy.kill_wait_s)
        pending = self.live_targets()
        self.record['cleanup']['remaining'] = list(pending.values())
        self.record['group_observation'].update(
            status='registered_targets_gone' if not pending else 'unknown', observed_at=now(),
            registered_pids=list(self.registry))
        self.record['cleanup']['status'] = ('failed' if self.record['errors']['cleanup_errors'] else
                                             'unverified' if pending else
                                             'completed' if self.primary else 'not_needed')

    def set_primary(self, exc, operation, stop_types):
        if self.primary is None:
            self.primary = exc
            self.record['errors']['primary_error'] = error(exc, operation)
            self.policy_stop = isinstance(exc, (StopRequested, KeyboardInterrupt, *stop_types))
            self.no_active_cleanup = isinstance(exc, TelemetryUnavailable)

    def fallback_record(self):
        try:
            self.fallback.write(json.dumps({'supervision_fallback': self.record}, ensure_ascii=False, default=str) + '\n')
            self.fallback.flush()
        except BaseException:
            pass


def execute_attempt(*, directory, command, cwd, session, code, config_ref,
                    stage, episode, owner_id, parent_attempt=None, parent_refs=None,
                    env=None, monitor_interval_s=1., wall_s=360., rss_bytes=2**31,
                    preflight=None, artifact=None, validate_artifact=None,
                    resource_check=None, postflight=None, evidence=None,
                    on_stop=None,
                    stop_types=(), backend=None, save=atomic_json, fallback=None):
    """Return a persisted passed terminal or raise SupervisionFailure(record, path).

    session supplies session_id, hard_stop and the six mandatory cleanup values.
    Admission is supplied as preflight; it runs after started and before spawn.
    artifact is a read-only observation; validate_artifact must verify its contract.
    Backend/save injection exists for fault tests, never to extend production time.
    """
    backend = backend or ProcessBackend()
    owned_fallback = None
    if fallback is None:
        owned_fallback = os.fdopen(os.dup(2), 'w')
        fallback = owned_fallback
    try:
        attempt = _Attempt(directory, command, cwd, session, code, config_ref, stage,
                           episode, owner_id, parent_attempt, parent_refs or [], backend, save, fallback)
    except BaseException as exc:
        try:
            fallback.write(json.dumps({'supervision_admission_denied': error(exc, 'receipt_directory_preflight')}, ensure_ascii=False) + '\n')
            fallback.flush()
        finally:
            if owned_fallback:
                owned_fallback.close()
        raise
    record = attempt.record
    streams = []
    try:
        # Real allocated bytes, not a sparse truncate. Released for terminal serialization.
        with attempt.reserve.open('xb') as reserve:
            reserve.write(b' ' * 262144)
            reserve.flush()
            os.fsync(reserve.fileno())
        try:
            save(attempt.start_path, record, True)
        except BaseException as exc:
            record['errors']['receipt_errors'].append(error(exc, 'started_receipt'))
            raise
        assert_prior_attempts_closed(directory, record['attempt_id'])
        if not owner_id or not session.get('session_id'):
            raise ValueError('explicit owner_id and session_id required')
        attempt.policy = Policy.from_session(session)
        stop_at = datetime.datetime.fromisoformat(session['hard_stop'])
        if stop_at.tzinfo is None:
            raise ValueError('hard_stop must contain timezone')
        attempt.hard_deadline = backend.monotonic() + stop_at.timestamp() - backend.wall_time()
        if attempt.hard_deadline - backend.monotonic() <= attempt.policy.cleanup_total_s:
            raise ValueError('insufficient hard_stop cleanup headroom')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
               for x in (monitor_interval_s, wall_s, rss_bytes)):
            raise ValueError('positive finite monitor, wall and RSS limits required')
        if preflight:
            preflight()
        if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            raise ValueError('default SIGCHLD required for exclusive Popen reaping ownership')
        record['prior_worker_artifact'] = artifact() if artifact else None
        finalizing = False
        def interrupt(signum, _frame):
            if finalizing:
                attempt.event('interrupt_during_cleanup', 'deferred', os.getpid())
                return
            raise StopRequested('supervisor_interrupt:' + str(signum))
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous = signal.getsignal(sig)
            signal.signal(sig, interrupt)
            attempt.handlers.append((sig, previous))
        for suffix in ('.stdout.log', '.stderr.log', '.resources.jsonl'):
            streams.append(attempt.prefix.with_suffix(suffix).open('xb'))
        if attempt.hard_deadline - backend.monotonic() <= attempt.policy.cleanup_total_s:
            raise ValueError('preflight consumed hard_stop cleanup headroom')
        attempt.proc = backend.spawn(command, cwd=cwd, env=env, stdout=streams[0],
                                     stderr=streams[1], start_new_session=True)
        record['child_exit']['status'] = 'running'
        record['child_exit']['pid'] = attempt.proc.pid
        deadline = min(attempt.started + wall_s, attempt.hard_deadline - attempt.policy.cleanup_total_s)
        completion_deadline = None
        retries = 0
        retry_deadline = None
        while attempt.poll() is None:
            instant = backend.monotonic()
            if instant >= deadline:
                raise StopRequested('stage_wall_limit_or_cleanup_headroom')
            try:
                resident = attempt.sample()
            except TelemetryUnavailable as exc:
                attempt.event('telemetry_retry', 'unavailable', attempt.proc.pid, exc)
                retry_deadline = retry_deadline or min(deadline, instant + attempt.policy.telemetry_retry_total_s)
                if attempt.wait(min(.02, max(0, retry_deadline - backend.monotonic())), 'Popen.telemetry_wait'):
                    break
                if retries >= attempt.policy.telemetry_retry_count or backend.monotonic() >= retry_deadline:
                    record['telemetry']['status'] = 'failed'
                    raise
                retries += 1
                backend.sleep(min(.02, max(0, retry_deadline - backend.monotonic())))
                continue
            retries, retry_deadline = 0, None
            if resident > rss_bytes:
                raise StopRequested('process_tree_rss_limit')
            if resource_check:
                resource_check(instant - attempt.started, resident)
            if artifact:
                record['worker_artifact'] = artifact()
                if record['worker_artifact']['status'] == 'failed':
                    raise RuntimeError('worker artifact reports failure')
                if record['worker_artifact']['status'] == 'completed' and completion_deadline is None:
                    completion_deadline = min(deadline, instant + attempt.policy.exit_grace_s)
            if completion_deadline is not None and backend.monotonic() >= completion_deadline:
                raise StopRequested('completed_worker_exit_grace_expired')
            streams[2].write((json.dumps({'elapsed_s': instant - attempt.started,
                                         'tree_rss_bytes': resident, 'observed_at': now(),
                                         'registered_targets': list(attempt.registry.values())}) + '\n').encode())
            streams[2].flush()
            backend.sleep(min(monitor_interval_s, max(0, deadline - backend.monotonic()),
                              max(0, completion_deadline - backend.monotonic()) if completion_deadline is not None else monitor_interval_s))
    except BaseException as exc:
        attempt.set_primary(exc, 'supervision_control', stop_types)
    finally:
        finalizing = True
        if on_stop and attempt.policy_stop:
            try:
                on_stop(attempt.primary)
            except BaseException as exc:
                record['errors']['evidence_errors'].append(error(exc, 'resource_stop_latch'))
        if attempt.proc and attempt.policy:
            try:
                attempt.cleanup()
            except BaseException as exc:
                record['errors']['cleanup_errors'].append(error(exc, 'cleanup'))
                record['cleanup']['status'] = 'failed'
        for stream in streams:
            try:
                stream.close()
            except BaseException as exc:
                record['errors']['evidence_errors'].append(error(exc, 'close_log'))
        if attempt.proc and record['child_exit']['status'] == 'running':
            record['child_exit']['status'] = 'unknown'
        if attempt.proc and record['reap']['status'] != 'reaped_by_popen':
            record['reap']['status'] = 'unverified'
        if artifact:
            try:
                record['worker_artifact'] = artifact()
                if record['worker_artifact']['status'] == 'completed' and validate_artifact:
                    if record['worker_artifact'].get('refs') == (record.get('prior_worker_artifact') or {}).get('refs'):
                        raise ValueError('worker artifact did not change in this attempt')
                    validate_artifact(record['worker_artifact'])
                    record['worker_artifact']['contract'] = 'passed'
                elif attempt.primary is None:
                    raise ValueError('completed, validated worker artifact required')
            except BaseException as exc:
                if attempt.primary is None:
                    attempt.set_primary(exc, 'worker_artifact_contract', stop_types)
                else:
                    record['errors']['evidence_errors'].append(error(exc, 'worker_artifact_contract'))
        elif attempt.primary is None:
            attempt.set_primary(ValueError('worker artifact contract callback required'), 'worker_artifact_contract', stop_types)
        if attempt.primary is None and record['child_exit']['returncode'] != 0:
            attempt.set_primary(RuntimeError('worker exit was not zero'), 'child_exit', stop_types)
        for callback, operation in ((postflight, 'postflight'), (evidence, 'top_level_evidence')):
            if callback:
                try:
                    callback()
                except BaseException as exc:
                    record['errors']['evidence_errors'].append(error(exc, operation))
        for sig, previous in reversed(attempt.handlers):
            try:
                signal.signal(sig, previous)
            except BaseException as exc:
                record['errors']['handler_errors'].append(error(exc, 'restore_signal_handler:' + str(sig)))
        record['registered_targets'] = list(attempt.registry.values())
        record['elapsed_s'] = backend.monotonic() - attempt.started
        record['finished_at'] = now()
        record['phase'] = 'terminal'
        ancillary = any(record['errors'][key] for key in ERROR_LISTS)
        closed = (not attempt.proc or record['reap']['status'] == 'reaped_by_popen') and record['cleanup']['status'] in ('not_needed', 'completed')
        record['supervisor_outcome'] = ('stopped' if attempt.policy_stop and closed and not ancillary else
                                        'passed' if attempt.primary is None and closed and not ancillary and record['worker_artifact'].get('contract') == 'passed' else 'failed')
        try:
            attempt.reserve.unlink(missing_ok=True)
            save(attempt.terminal, record, True)
        except BaseException as exc:
            record['errors']['receipt_errors'].append(error(exc, 'terminal_receipt'))
            record['supervisor_outcome'] = 'failed'
            attempt.fallback_record()
        if owned_fallback:
            owned_fallback.close()
    if record['supervisor_outcome'] != 'passed':
        raise SupervisionFailure(record, attempt.terminal) from attempt.primary
    return record


@contextmanager
def control_scope(owner, prefix, *, metadata=None, evidence=None, save=atomic_json, fallback=None):
    """Protect legacy Owner exit, descriptor release and top-level evidence separately.

    Uses Owner's existing kernel lock and history. It does not change SES admission.
    Caller body errors survive as primary_error even if Owner.__exit__ also fails.
    """
    prefix = Path(prefix)
    value = {'schema': 'r12_controller_finalization_v1', 'phase': 'started',
             'created_at': now(), 'metadata': metadata or {}, 'errors': errors(),
             'supervisor_outcome': 'unknown'}
    primary = None
    entered = False
    owned_fallback = os.fdopen(os.dup(2), 'w') if fallback is None else None
    fallback = fallback or owned_fallback
    try:
        save(prefix.with_suffix('.started.json'), value, True)
        current = owner.__enter__()
        entered = True
        yield current
    except BaseException as exc:
        primary = exc
        value['errors']['primary_error'] = error(exc, 'controller_body_or_owner_enter')
    finally:
        # Evidence can update the shared ledger/index. Keep the acquired Owner
        # lock through that callback; only the immutable control receipt follows exit.
        if evidence and entered:
            try:
                evidence()
            except BaseException as exc:
                value['errors']['evidence_errors'].append(error(exc, 'controller_evidence'))
        if entered:
            try:
                owner.__exit__(type(primary) if primary else None, primary,
                               primary.__traceback__ if primary else None)
            except BaseException as exc:
                value['errors']['owner_errors'].append(error(exc, 'Owner.__exit__'))
        fd = getattr(owner, 'fd', None)
        if fd is not None and not fd.closed:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BaseException as exc:
                value['errors']['owner_errors'].append(error(exc, 'Owner.lock_release'))
            try:
                fd.close()
            except BaseException as exc:
                value['errors']['owner_errors'].append(error(exc, 'Owner.fd_close'))
        value.update(phase='terminal', finished_at=now(),
                     supervisor_outcome='failed' if primary or any(value['errors'][key] for key in ERROR_LISTS) else 'passed')
        try:
            save(prefix.with_suffix('.json'), value, True)
        except BaseException as exc:
            value['errors']['receipt_errors'].append(error(exc, 'controller_terminal_receipt'))
            value['supervisor_outcome'] = 'failed'
            try:
                fallback.write(json.dumps({'controller_fallback': value}, ensure_ascii=False) + '\n')
                fallback.flush()
            except BaseException:
                pass
        if owned_fallback:
            owned_fallback.close()
    if value['supervisor_outcome'] != 'passed':
        raise SupervisionFailure(value, prefix.with_suffix('.json')) from primary
