# System Architecture — Instrumented VQC Pipeline

> **Status:** Proposed (DAG-validated, not yet implemented)
> **Owner:** Cristóbal Cheuquel, Valentina Huenchuñir
> **Last update:** 2026-07-05

## Goal

Add progressive persistence, traceability, and graceful shutdown to the existing VQC pipeline at `/home/cris/Workspace/26-1/cuantica/ElectivoCuantica/`. The architecture is backend-agnostic: identical instrumentation for `statevector`, `spinq_nmr`, `aer_simulator`, `ibm_simulator`, `ibm_hardware`.

## High-level component diagram

```mermaid
flowchart TB
    subgraph CLI["main.py — entry point"]
        A[parse env: BACKEND_MODE, MAX_ITER, N_SHOTS]
        B[init_run: run_id, dirs, JSONL writer]
        C[install signal handlers: SIGINT, SIGTERM, SIGALRM]
        D[start heartbeat thread: 30s]
    end

    subgraph INSTR["Instrumentation layer (NEW)"]
        E[JsonlEventLogger]
        F[StateSnapshot — derived from JSONL]
        G[SignalHandler: graceful shutdown]
        H[Watchdog: reset on progress]
    end

    subgraph PREF["Pre-flight"]
        I[verify SDK importable]
        J[connect to backend]
        K[calibration circuit: shots_per_sec]
        L{preflight OK?}
    end

    subgraph CORE["Experimental core"]
        M[preprocess: load, filter, split, LDA, scale]
        N[for each ansatz: Base, Reducido, HEA]
        O[for each optimizer: COBYLA, SPSA]
        P[fit: loss per iter, flush to JSONL every iter]
        Q[eval: predictions, metrics]
    end

    subgraph BACK["Backends (same contract)"]
        R[StatevectorBackend]
        S[SpinQNMRBackend]
        T[AerBackend]
        U[IBMSimulatorBackend]
        V[IBMHardwareBackend]
    end

    subgraph PERSIST["Progressive persistence"]
        W[(events.jsonl — append-only, ordered)]
        X[(state.json — derived snapshot)]
        Y[(results.csv — append per ansatz)]
        Z[(predictions.csv — append per ansatz)]
        AA[(theta_ansatz.npz — per ansatz, every iter)]
    end

    A --> B --> C --> D --> I --> J --> K --> L
    L -->|no| AB[EMIT error event, EXIT 1, NO FALLBACK]
    L -->|yes| M --> N --> O --> P --> Q
    P -.flush.-> W
    P -.flush.-> AA
    P -.reset.-> H
    Q -.append.-> Y
    Q -.append.-> Z
    E --> W
    F --> X
    G -.trigger.-> F
    G -.trigger.-> W
    H -.trigger.-> AB
    R & S & T & U & V -.same expectations API.-> P
```

## Key invariants

- **NO fallback policy.** If preflight fails (SDK missing, backend unreachable, calibration hang), the run emits an error event, writes `state.json` with `status=failed`, and exits 1. Silent degradation to a different backend is forbidden.
- **JSONL is the single source of truth.** `state.json` is a derived snapshot; if it is corrupted, it can be reconstructed from `events.jsonl`.
- **Same schema for every backend.** `statevector`, `spinq_nmr`, `aer_simulator`, `ibm_simulator`, `ibm_hardware` all emit the same event types and the same state.json structure.
- **Per-iteration fsync on simulator, batched (5 iters or 30 s) on hardware.** Trade-off between robustness and throughput.
- **Phase-aware watchdog.** 300 s for simulator, 1800 s for `spinq_nmr`, 3600 s for `ibm_hardware`. Different phases (calibration, training) have different timeout multipliers.
```