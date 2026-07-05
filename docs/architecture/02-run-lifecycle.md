# Run Lifecycle

End-to-end sequence of a single `python main.py` invocation, from process start to exit.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant M as main.py
    participant L as JSONL events
    participant S as state.json
    participant B as Backend
    participant H as Heartbeat
    participant W as Watchdog

    User->>M: python main.py (BACKEND_MODE=spinq_nmr)
    M->>L: emit init {ts, run_id, config}
    M->>S: snapshot init
    M->>B: preflight.connect()
    B-->>M: ok | error
    alt error
        M->>L: emit error {reason, traceback, ts}
        M->>S: status=failed
        M-->>User: exit 1
    end
    M->>B: preflight.calibrate()
    B-->>M: shots_per_sec
    M->>L: emit preflight_done {shots_per_sec}
    M->>H: start (interval=30s)
    M->>W: start (timeout=300s/1800s/3600s, reset on iter)

    loop ansatz in [Base, Reducido, HEA]
        M->>L: emit ansatz_start {ansatz, n_params, theta_init_hash}
        loop iter in 1..MAX_ITER
            M->>B: loss(theta)
            B-->>M: value
            M->>L: emit iter {iter, loss, theta_hash, wall_ms}
            M->>S: append loss[i] to history
            M->>S: snapshot
            M->>W: reset
        end
        M->>B: evaluate(X_test)
        B-->>M: predictions, expectations
        M->>L: emit eval_end {accuracy, f1, ...}
        M->>Y: append row
        M->>Z: append rows
        M->>L: emit ansatz_end
    end

    M->>H: stop
    M->>W: stop
    M->>L: emit complete
    M->>S: status=completed
    M-->>User: exit 0
```

## Phase boundaries (state.json snapshots)

A `state.json` snapshot is written (atomically) at these events:
- `init`
- `preflight_ok`
- `calibration_done`
- `ansatz_end` of each ansatz
- `eval_end` of each ansatz
- `shutdown` (any reason: user interrupt, watchdog, normal exit)

The source of truth is `events.jsonl`. `state.json` is a derived view for cheap reading and resume logic.
```