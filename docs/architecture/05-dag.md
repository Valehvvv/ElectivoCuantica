# Task DAG — 4 Phases, 22 Atomic Tasks

```mermaid
graph LR
  subgraph P1["Phase 1: Foundation"]
    P1T1[schemas] --> P1T2[validators]
    P1T2 --> P1T3[preflight]
    P1T3 --> P1T4[calibration]
    P1T4 --> P1T5[backend contract]
    P1T5 --> P1T6[migrate]
    P1T6 --> P1T7[skeleton runner + --pilot]
  end
  
  subgraph P2["Phase 2: Persistence"]
    P1T2 --> P2T1[JsonlEventLogger]
    P2T1 --> P2T2[StateSnapshot]
    P2T2 --> P2T3[theta persist]
    P2T3 --> P2T4[CSV append]
    P2T4 --> P2T5[iteration loop]
    P1T7 --> P2T5
  end
  
  subgraph P3["Phase 3: Resilience"]
    P2T1 --> P3T1[signal handler]
    P3T1 --> P3T2[watchdog phase-aware]
    P3T2 --> P3T3[heartbeat 30s]
    P3T3 --> P3T4[graceful shutdown]
    P3T4 --> P3T5[integrate stack]
    P2T5 --> P3T5
  end
  
  subgraph P4["Phase 4: Optimizers + Pilot"]
    P2T5 --> P4T1[SPSA]
    P3T5 --> P4T1
    P4T1 --> P4T2[pilot flag]
    P4T2 --> P4T3[methodology.md]
    P4T3 --> P4T4[README]
  end
```

## PR boundaries

- **PR #1** (Phase 1): runnable skeleton, no training yet. Validation: `python main.py --backend spinq_nmr` exits 1 with proper error event.
- **PR #2** (Phase 2): training loop survives SIGKILL mid-iter. Validation: stress test with `os.kill(os.getpid(), signal.SIGKILL)` after N iters.
- **PR #3** (Phase 3): signal/watchdog/heartbeat integration. Validation: `os.kill(os.getpid(), signal.SIGINT)` after 2 iters → state shows `status=interrupted`, `current_iter=2`.
- **PR #4** (Phase 4): both optimizers + pilot. Validation: `--pilot` produces run with `MAX_ITER=5, N_SHOTS=64`.

## Commit sequence (22 commits, 4 PRs)

Each task is one commit. PRs bundle commits in their phase.
```