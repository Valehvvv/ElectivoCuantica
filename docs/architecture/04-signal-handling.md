# Signal Handling & Shutdown

```plantuml
@startuml
title Run lifecycle with robust shutdown

[*] --> Init
Init --> Preflight
Preflight --> PreflightFailed : error / SDK no importable / backend unreachable
PreflightFailed --> [*] : emit error event, exit 1

Preflight --> Training : preflight ok, calibration ok
Training --> HeartbeatLoop : start 30s
Training --> WatchdogLoop : start phase-aware timeout

Training --> IterRunning : optimizer step
IterRunning --> IterRunning : loss computed, emit iter event, fsync, reset watchdog
IterRunning --> AnsatzDone : iter == MAX_ITER
AnsatzDone --> Evaluation : evaluate on test set
Evaluation --> AnsatzLogged : append to results.csv, predictions.csv, emit eval event
AnsatzLogged --> Training : next ansatz
AnsatzLogged --> [*] : all ansatze done, emit complete, exit 0

state "External signal" as Signal
Training --> Signal : SIGINT / SIGTERM
IterRunning --> Signal : SIGINT / SIGTERM
AnsatzLogged --> Signal : SIGINT / SIGTERM
Signal --> Dumping : set interrupted=True
Dumping --> [*] : emit signal+shutdown, status=interrupted, write state, exit 130

WatchdogLoop --> Dumping : N seconds without progress
Dumping --> [*] : reason=watchdog_timeout, status=interrupted

note right of Dumping
  Atomic dump:
  1. fsync events.jsonl
  2. snapshot state.json
  3. fsync state.json
  4. emit shutdown event
end note

@enduml
```

## Pattern: O_NONBLOCK self-pipe

Python signal handlers must be async-signal-safe. The only safe operations in a handler are:
- write to an `O_NONBLOCK` pipe (1 byte)
- set an atomic flag (`_thread.interrupt_flag` or `signal.set_wakeup_fd`)

Heavy work (flushing JSONL, writing state.json) is done in the main thread, on the next loop iteration, after the handler returns.

```python
import os, signal, select

_read_fd, _write_fd = os.pipe()
os.set_blocking(_write_fd, False)  # O_NONBLOCK
os.set_blocking(_read_fd, False)

def _handler(signum, frame):
    try:
        os.write(_write_fd, b"\x00")  # 1 byte, ignore EAGAIN
    except BlockingIOError:
        pass  # pipe full, signal already pending

signal.signal(signal.SIGINT, _handler)
signal.signal(signal.SIGTERM, _handler)
```

Main loop uses `select.select([_read_fd], [], [], 0.1)` to detect signal without busy-waiting.

## Double-signal escalation

A second SIGINT within 2 s forces immediate exit (no flush). This is the user's "really kill it" path.
```