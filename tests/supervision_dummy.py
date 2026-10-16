"""Finite, model-free fixtures. Every spawned descendant belongs to this fixture."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time


def main():
    mode, raw_root = sys.argv[1:3]
    root = Path(raw_root)
    root.mkdir(parents=True, exist_ok=True)
    if mode == 'crash-controller':
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from r12_supervision import ProcessBackend, execute_attempt
        import datetime
        class CrashBackend(ProcessBackend):
            def spawn(self, command, **kwargs):
                child = super().spawn(command, **kwargs)
                started=json.loads(next((root/'attempts').glob('*.started.json')).read_text())
                identity=self.identity(child.pid,started['owner_id'],started['attempt_id'])
                (root/'crash_child_pid.json').write_text(json.dumps(identity))
                os._exit(77)  # Simulated controller crash after durable started + spawn.
        execute_attempt(directory=root/'attempts', command=[sys.executable, '-B', __file__, 'natural', str(root/'child')],
                        cwd=root, session={'session_id': 'crash-fixture',
                            'hard_stop': (datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=10)).isoformat(),
                            'cleanup': {'exit_grace_s': .3, 'term_wait_s': .3, 'kill_wait_s': .3,
                                        'cleanup_total_s': 1.5, 'telemetry_retry_count': 2, 'telemetry_retry_total_s': .2}},
                        code={'commit': 'fixture', 'tree': 'fixture', 'dirty': False}, config_ref=None,
                        stage='dummy', episode=None, owner_id='crash-fixture-owner', backend=CrashBackend())
        return
    (root/(mode+'_pid.json')).write_text(json.dumps({'pid': os.getpid(), 'ppid': os.getppid()}))
    if mode in ('natural', 'late'):
        (root/'completed.json').write_text(json.dumps({'mode': mode, 'pid': os.getpid()}))
        time.sleep(.12 if mode == 'late' else .04)
        return
    if mode == 'nonzero':
        (root/'completed.json').write_text('{"status":"completed"}')
        sys.exit(7)
    if mode == 'ignore':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (root/'ready').write_text('ready')
        time.sleep(3.)
        return
    if mode in ('leaf', 'term', 'orphan'):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        (root/(mode+'_ready')).write_text('ready')
        time.sleep(.22 if mode == 'orphan' else 3.)
        return
    if mode in ('tree', 'branch', 'leader-first'):
        childmode = 'branch' if mode == 'tree' else 'orphan' if mode == 'leader-first' else 'leaf'
        child = subprocess.Popen([sys.executable, '-B', __file__, childmode, str(root)])
        def finish(*_):
            # Supervisor sends descendants first. This parent is their sole reaper.
            try:
                child.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                pass
            sys.exit(0)
        signal.signal(signal.SIGTERM, finish)
        end = time.monotonic() + 3.
        while time.monotonic() < end:
            if mode == 'leader-first' and (root/'registered_ack').exists():
                (root/'completed.json').write_text('{"status":"completed"}')
                return  # Orphan exits on its own; no signal to a reparented identity.
            time.sleep(.01)
        child.wait(timeout=1.)
        return
    raise ValueError(mode)


if __name__ == '__main__':
    main()
