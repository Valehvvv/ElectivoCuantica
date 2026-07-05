# Per-iteration Instrumentation

The fit loop emits one event per loss evaluation. Atomicity is preserved at the backend-call boundary: if Ctrl+C arrives during the backend call, the current iter is either complete (event already written) or absent (no half-event).

```mermaid
sequenceDiagram
    autonumber
    participant Opt as Optimizer
    participant Lf as loss(theta)
    participant B as Backend
    participant E as EventLogger
    participant FS as Disk
    participant W as Watchdog

    Opt->>Lf: theta_i
    Lf->>B: build circuits (N_train)
    B-->>Lf: expectations[0..N_train]
    Lf->>Lf: compute BCE
    Lf-->>Opt: loss_i
    Lf->>E: event(LEVEL_INFO, "iter", iter=i, loss=..., theta_sha256=..., wall_ms=...)
    E->>FS: append 1 line to events.jsonl
    E->>FS: fsync (durability, batched on hardware)
    E->>W: watchdog.reset()
    Note over Lf,FS: If Ctrl+C here → 0 iters lost (current iter already persisted atomically)
    Note over Opt,E: If Ctrl+C mid-eval of backend → exception, signal handler catch in main thread
```

## Fsync strategy

| Backend | Fsync cadence | Reason |
|---|---|---|
| `statevector` | every iter | cheap, maximum durability |
| `aer_simulator` | every iter | cheap, maximum durability |
| `spinq_nmr` | every 5 iters OR 30 s wall-clock | per-iter fsync adds visible overhead on slow hardware |
| `ibm_simulator` | every 5 iters OR 30 s | same as above |
| `ibm_hardware` | every 5 iters OR 30 s | same as above |

`watchdog.reset()` is called on every loss emission regardless of fsync batch — the watchdog measures logical progress, not disk activity.

## Theta persistence

After every iter, the current `theta_<ansatz>.npz` is overwritten (atomic write via temp file + rename). On resume or post-mortem, the last known good theta is always loadable.
```