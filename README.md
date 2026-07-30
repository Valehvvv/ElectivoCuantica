# ElectivoCuantica

Proyecto semestral electivo ICC403-1: COMPUTACIÓN Y PROGRAMACIÓN CUÁNTICA (E/E)

Comparación de ansätze variacionales de 2 qubits para clasificación binaria
sobre el dataset Iris (Versicolor vs. Virginica), ejecutado en múltiples
plataformas cuánticas.

Esta rama (`main`) es la base mínima del repositorio (solo el notebook
original y este README). Toda la implementación vive en las ramas feature:

## Ramas del repositorio

- **`main`** — Rama base/mínima. Solo contiene el notebook original
  (`preprocessingfinalfinal.ipynb`) y este README. Punto de partida sin
  integraciones de backend.

### `feature/IBM-architecture`

Integración con IBM Quantum, con foco en observabilidad y control de costos
al ejecutar en QPU real. 126 archivos, +10321 líneas vs `main`.

**Flujo (`main.py`):**
1. Preprocesamiento: carga Iris, filtra par binario, LDA a 2D, normaliza a [0, π], subsample opcional (`N_TRAIN`).
2. Resolución de backend según `BACKEND_MODE` (`statevector`, `aer_simulator`, `ibm_simulator`, `ibm_hardware`, `spinq_nmr`). Si el backend falla, cae a `statevector`.
3. Selección de backend IBM (`ibm_runtime.py`): si hay `IBM_BACKEND_NAME` explícito lo usa; si `USE_IBM_SIMULATOR=True` toma el primer simulador; si no, llama `service.least_busy(operational=True, simulator=False, min_num_qubits=2)` para elegir la QPU real menos ocupada.
4. Entrenamiento por ansatz con COBYLA (scipy). Cada iteración envía **todos** los circuitos de la muestra en un solo job (`EstimatorV2.run(pubs)`), no un job por muestra — clave para no gastar cuota.
5. Evaluación, export de `results.csv`, `predictions.csv`, gráficos, y `run_metadata.json` (con redacción de credenciales, hash de commit, log de uso IBM).

**Observabilidad y guardrails de costo:**
- Logger dual: consola (coloreado, TTY-aware) + archivo JSON Lines (`logs/<run_id>_<backend>.log`), captura DEBUG+.
- Preflight check antes de cargar datos: verifica que el SDK del backend esté importable, falla rápido si falta.
- Timeout por job IBM: `IBM_MAX_EXECUTION_TIME` (default 60s).
- Watchdog con timeout por fase: 300s (simulador local), 1800s (SpinQ), 3600s (IBM). Lanza `TimeoutError` y dispara shutdown ordenado.
- Heartbeat cada 30s (thread daemon) para detectar runs colgados.
- Manejo de señales (`SelfPipeSignal`, self-pipe trick) captura SIGINT/SIGTERM incluso durante llamadas bloqueantes al backend.
- **Nota:** no hay cap de costo en dólares/créditos explícito, solo timeout de tiempo de ejecución.

**Persistencia y recuperación de estado:**
- `events.jsonl` es la fuente de verdad (append-only, un evento JSON por línea: `init`, `preflight_ok/failed`, `iter_start/complete/error`, `shutdown`, `complete`, `error`).
- `state.json` se reconstruye desde `events.jsonl` vía `rebuild_state()` — es un artefacto derivado, no la fuente.
- Si un run se interrumpe, `events.jsonl` queda válido hasta el último evento completado; la recuperación es manual (releer eventos + rebuild), no hay resume automático (para evitar corromper parámetros a medias).

**Variables de entorno propias:** `IBMQ_TOKEN`, `IBMQ_INSTANCE`, `IBM_MAX_EXECUTION_TIME`.

### `feature/refactor-vqc-ibm`

Refactor del VQC que añade QRydDemo (átomos de Rydberg) y corrige el bug de
endianness de SpinQ NMR. 127 archivos, +10477 líneas vs `main`. Comparte la
misma infraestructura de persistencia/resiliencia/observabilidad que
`feature/IBM-architecture` (event-sourced logging, watchdog, heartbeat,
graceful shutdown) — la diferencia está en los backends soportados.

**Flujo:** mismo esqueleto que la rama IBM (preprocesamiento → resolución de
backend → entrenamiento batched por ansatz con COBYLA → evaluación → export),
pero con dos backends adicionales:

**QRydDemo (`qryd_backend.py`):**
- Backend cloud de simulación de átomos de Rydberg (`QRydProvider`).
- Cuota diaria de 5000 unidades: si `BACKEND_MODE=qryd`, el pipeline reduce automáticamente `MAX_ITER` a 15 y `N_TRAIN` a 10 (salvo que se sobreescriban por env) para no exceder el límite en una corrida completa (3 ansätze).
- Backend seleccionable vía `QRYD_BACKEND_NAME` (default `qryd_emulator$square`).

**SpinQ NMR (`spinq_backend.py`) — fix de endianness:**
- Bug encontrado: interpretar el bitstring de resultados como little-endian (convención Qiskit/IBM) daba 0.00 de accuracy en SpinQ; cambiar a **big-endian** (bit 0 del string = qubit 0, no el qubit de mayor índice) subió a 0.35. Es la única diferencia entre ambas interpretaciones.
- **Nota abierta:** este fix no se verificó todavía contra hardware SpinQ real (sin acceso al armar el análisis); forma de verificar: aplicar X al qubit 1 vía Ry y ver qué carácter del bitstring cambia.
- Traducción de circuito Qiskit → formato nativo SpinQ (`_qiskit_to_spinq`): `Ry` y `CX` se mapean directo; `ECR` se descompone como `H · CX · H` (aproximación).
- `N_TEST` (nuevo): controla tamaño de subsample de test igual que `N_TRAIN` controla el de train; para SpinQ ambos se autorreducen (`N_TRAIN=3`, `N_TEST=3`, `MAX_ITER=1`) porque cada circuito tarda ~7s entrenar / ~35s evaluar en el hardware real de la universidad, y sin esto una corrida no entra en ~15 min.
- Requiere entorno separado `.venv-spinq` (Python 3.9, incompatible con el resto del proyecto en 3.12+); `check_spinq_environment()` verifica que `spinqit` sea importable y avisa con instrucciones (`scripts/setup_spinq.sh`) si no, sin fallar silenciosamente.

**Backend abstraction compartida (`backends.py`):** todos los backends implementan `expectations(circuits) -> list[float]`; el observable medido es Z sobre el qubit físico de índice 1 (`observable.MEASURED_QUBIT_INDEX`), con la tabla de endianness por backend (`statevector`/`aer`/`ibm` = little, `spinq_nmr` = big, `qryd` = little) resuelta en `observable.expectation_z_qubit0_from_counts()`.

**Variables de entorno propias:** `N_TEST`, `QRYD_API_TOKEN`, `QRYD_BACKEND_NAME`, `SPINQ_IP`, `SPINQ_PORT`, `SPINQ_USERNAME`, `SPINQ_PASSWORD`, `SPINQ_TASK_NAME`.

Ambas ramas feature construyen infraestructura casi idéntica (abstracción de
backends, persistencia, resiliencia, schemas, tests) pero divergen en
integraciones específicas como se detalla arriba.

## Notebook original

El notebook `preprocessingfinalfinal.ipynb` se mantiene sin modificar como
referencia del trabajo original.
