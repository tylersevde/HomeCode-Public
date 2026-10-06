"""Narrow privileged governor lease. No command execution or arbitrary sysfs paths."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import struct
import time

GOVERNOR = Path('/sys/devices/system/cpu/cpufreq/policy0/scaling_governor')
ALLOWED = ('ondemand','performance')

class Lease:
    def __init__(self, read, write, alive, now, seconds):
        self.read,self.write,self.alive,self.now=read,write,alive,now
        self.original=read();self.deadline=now()+seconds;self.last=now();self.closed=False
        self.restoration_reason=None;self.restored_monotonic=None
    def check(self):
        if self.closed:return False
        if not self.alive():reason='controller_unavailable'
        elif self.now()>=self.deadline:reason='deadline'
        elif self.now()-self.last>60:reason='heartbeat_timeout'
        else:return True
        self.restore(reason);return False
    def request(self, desired=None):
        if self.closed or not self.check():raise RuntimeError('Governor lease expired')
        if desired is not None:
            if desired not in ALLOWED:raise ValueError('Governor is outside frozen comparison')
            self.write(desired)
            if self.read()!=desired:raise RuntimeError('Governor readback failed')
        self.last=self.now()
        return dict(original=self.original,current=self.read(),deadline=self.deadline)
    def restore(self, reason='explicit_close'):
        # Cleanup may retry restoration after an earlier failure or closed request.
        # Keep the initiating reason and the first successful restoration time.
        if self.restoration_reason is None:self.restoration_reason=reason
        self.write(self.original)
        if self.read()!=self.original:raise RuntimeError('Governor restoration failed')
        self.closed=True
        if self.restored_monotonic is None:self.restored_monotonic=self.now()

def controller_start(process):
    """Read Linux start ticks without splitting the possibly spaced comm field."""
    try:
        text=process.read_text()
        prefix,separator,tail=text.rpartition(')')
        if not separator or '(' not in prefix:return None
        fields=tail.split()
        if fields[0] in ('Z','X','x'):return None
        return str(int(fields[19]))
    except (OSError,ValueError,IndexError):return None

def log_event(event, controller, **values):
    print(json.dumps(dict(event=event,utc=datetime.now(timezone.utc).isoformat(),
        monotonic=time.monotonic(),controller=controller,**values)),flush=True)

def request(path, desired=None, close=False):
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as conn:
        conn.settimeout(3);conn.connect(str(path))
        conn.sendall(json.dumps(dict(desired=desired,close=close)).encode()+b'\n')
        data=b''
        while not data.endswith(b'\n'):
            chunk=conn.recv(4096)
            if not chunk:raise RuntimeError('Governor helper disconnected')
            data+=chunk
        value=json.loads(data)
        if 'error' in value:raise RuntimeError(value['error'])
        return value

def serve(lease, path, uid, controller):
    """Serve a lease; injectable state also permits lifecycle tests without sysfs."""
    exit_reason='helper_exit'
    def stopping(signum, _frame):
        nonlocal exit_reason
        if exit_reason.startswith('signal_'):return
        exit_reason='signal_'+signal.Signals(signum).name
        raise SystemExit('Governor helper stopped')
    old_handlers={s:signal.signal(s,stopping) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
            server.bind(str(path));os.chown(path,uid,-1);os.chmod(path,0o600);server.listen(2);server.settimeout(1)
            log_event('ready',controller,original=lease.original,pid=os.getpid(),deadline=lease.deadline)
            while lease.check() and not lease.closed:
                try:conn,_=server.accept()
                except socket.timeout:continue
                with conn:
                    conn.settimeout(2)
                    try:
                        peer=struct.unpack('3i',conn.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
                        if peer[1]!=uid:raise ValueError('Peer owner mismatch')
                        data=b''
                        while not data.endswith(b'\n') and len(data)<4096:
                            chunk=conn.recv(4096)
                            if not chunk:raise ValueError('Incomplete request')
                            data+=chunk
                        req=json.loads(data)
                        if req.get('close'):
                            lease.restore('explicit_close');value=dict(original=lease.original,current=lease.read(),restored=True)
                        else:value=lease.request(req.get('desired'))
                        log_event('request',controller,**value)
                    except Exception as exc:value=dict(error=str(exc))
                    conn.sendall(json.dumps(value).encode()+b'\n')
    except Exception as exc:
        exit_reason='helper_error'
        log_event('error',controller,error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            lease.restore(exit_reason)
            log_event('restored',controller,original=lease.original,current=lease.read(),
                reason=lease.restoration_reason,restored_monotonic=lease.restored_monotonic,
                last_heartbeat_monotonic=lease.last,deadline=lease.deadline)
        except Exception as exc:
            log_event('restoration_failed',controller,reason=lease.restoration_reason,
                error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            try:path.unlink(missing_ok=True)
            finally:
                for signum,handler in old_handlers.items():signal.signal(signum,handler)

def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);p.add_argument('--pid',type=int,required=True)
    p.add_argument('--seconds',type=float,required=True);a=p.parse_args()
    uid=int(os.environ['PKEXEC_UID'])
    if os.geteuid()!=0 or not 120<a.seconds<=14400:raise ValueError('Invalid privileged lease')
    process=Path(f'/proc/{a.pid}/stat');identity=controller_start(process)
    if identity is None:raise ValueError('Controller is not live')
    if process.stat().st_uid!=uid:raise ValueError('Controller owner mismatch')
    controller=dict(pid=a.pid,start_ticks=identity,uid=uid)
    lease=Lease(lambda:GOVERNOR.read_text().strip(),lambda v:GOVERNOR.write_text(v),
        lambda:controller_start(process)==identity,time.monotonic,a.seconds)
    path=Path(a.socket)
    if path.exists() or not path.is_absolute() or path.parent.stat().st_uid!=uid:raise ValueError('Invalid socket destination')
    serve(lease,path,uid,controller)

if __name__=='__main__':main()
