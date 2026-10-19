"""One foreground local supervisor: real child PID, append log, durable exit.

Run in a new task-owned Terminal. No service installation and no restart loop.
The child inherits the execution lock, so a surviving child prevents a duplicate
launch even if this supervisor dies. SIGKILL cannot produce an exit receipt;
absence of that receipt is never evidence that the worker is alive.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, state):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(state,stream,sort_keys=True); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir',type=Path,required=True)
    parser.add_argument('--progress',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(); command=args.command
    if command[:1]==['--']: command=command[1:]
    if not command: parser.error('worker command required after --')
    root=args.state_dir.resolve(); root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Existing supervisor/worker holds execution lock; no launch')
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+str(os.getpid())
        receipt=root/(run_id+'.json'); log_path=root/'worker.log'
        state={'run_id':run_id,'supervisor_pid':os.getpid(),'supervisor_ppid':os.getppid(),
               'started_at':now(),'command':command,'log':str(log_path),'status':'starting'}
        save(receipt,state)
        child=None
        def forward(signum, frame):
            state['supervisor_signal']=signum
            save(receipt,state)
            if child is not None and child.poll() is None: child.send_signal(signum)
        for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(signum,forward)
        with log_path.open('a',buffering=1) as log:
            log.write(json.dumps({'supervisor_start':state})+'\n'); log.flush()
            try:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                       pass_fds=(lock.fileno(),),start_new_session=True)
                state.update(child_pid=child.pid,child_parent_pid=os.getpid(),status='running')
                save(receipt,state); save(root/'current.json',state)
                print(json.dumps(state),flush=True)
                while child.poll() is None:
                    state.update(observed_at=now(),disk_free_bytes=shutil.disk_usage(root).free)
                    try:
                        state['progress']=json.loads(args.progress.read_text())
                        state['progress_mtime_ns']=args.progress.stat().st_mtime_ns
                    except (OSError,ValueError): pass
                    # A single exact-PID resource read; failure does not imply death.
                    try:
                        usage=subprocess.run(['/bin/ps','-p',str(child.pid),'-o','rss='],capture_output=True,text=True)
                        if usage.returncode==0 and usage.stdout.strip().isdigit(): state['child_rss_kib']=int(usage.stdout.strip())
                    except OSError as exc:
                        state['resource_probe_error']=str(exc)
                    save(receipt,state); save(root/'current.json',state)
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: pass
                state.update(status='finished',returncode=child.returncode,finished_at=now())
            except BaseException as exc:
                state.update(status='supervisor_error',error=repr(exc),observed_at=now())
                if child is not None and child.poll() is not None: state['returncode']=child.returncode
                raise
            finally:
                save(receipt,state); save(root/'current.json',state)
                log.write(json.dumps({'supervisor_end':state})+'\n'); log.flush(); os.fsync(log.fileno())
        return child.returncode


if __name__=='__main__':
    raise SystemExit(main())
