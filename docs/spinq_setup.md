# SpinQ NMR: entorno dedicado (Fase 5)

## Por qué un entorno separado

El proyecto principal (`pyproject.toml`, `uv.lock`) usa **Python >= 3.12**
y `qiskit >= 2.5.0` / `numpy >= 2.5.0`. El SDK de SpinQ (`spinqit`) **sí
está publicado en PyPI**, pero **solo con wheels para cp38/cp39/cp310**
(no hay wheels para 3.11+), por lo que **no puede resolverse** en el
entorno principal 3.12 (`uv add spinqit` falla ahí con "no wheels with
matching Python version tag cp312"). Por eso `spinqit` necesita un venv
Python 3.9 **dedicado y separado** del `.venv` principal.

**Prerequisito:** estar en la misma LAN que el computador NMR, y tener la
IP + credenciales del dispositivo configuradas en `.env` (`SPINQ_IP`,
`SPINQ_PORT`, `SPINQ_USERNAME`, `SPINQ_PASSWORD`, `SPINQ_TASK_NAME`).

**Solo el modo `BACKEND_MODE = "spinq_nmr"` necesita el entorno 3.9.**
Todo lo demás del proyecto (`statevector`, `aer_simulator`, `ibm_simulator`,
`ibm_hardware`, la suite de tests, el pipeline clásico) sigue corriendo
normalmente en el entorno principal (Python 3.12, `uv sync` / `uv run`)
**sin ningún cambio** -- los dos entornos coexisten.

## Cómo se separaron las dependencias

- `pyproject.toml` declara un extra **informativo** `[project.optional-dependencies] spinq`
  que lista únicamente los paquetes ya presentes en PyPI que
  `spinq_backend.py` necesita además de `spinqit` (`numpy`, `qiskit`), sin
  fijar versiones nuevas — por eso `uv lock`/`uv sync` en el entorno
  principal no cambian de resultado exista o no ese extra, y por eso
  **no** se instala por defecto (`uv sync` normal lo ignora).
- Ese extra es solo documentación de "qué falta aparte de spinqit"; **no
  uses** `uv sync --extra spinq` en el `.venv` principal — `spinqit`
  seguiría sin resolverse ahí porque el entorno principal es 3.12 y
  `spinqit` en PyPI solo publica wheels cp38/cp39/cp310. El entorno 3.9
  real se crea aparte, con `scripts/setup_spinq.sh`, sin tocar el
  `uv.lock` del proyecto principal.
- `spinq_backend.py` sigue haciendo el `import spinqit` de forma
  **perezosa** (dentro de `connect()`), tal como antes de la Fase 5, de
  modo que **importar el módulo en Python 3.12 nunca falla** aunque
  `spinqit` no esté instalado.
- Se añadió un *guard* de entorno (`spinq_backend.check_spinq_environment`,
  ver más abajo) que se ejecuta antes de intentar `import spinqit`, para
  dar un mensaje accionable en vez de un `ImportError` genérico o un
  fallo silencioso.

## Setup: `scripts/setup_spinq.sh`

Desde la raíz del proyecto (`ElectivoCuantica/`), basta con ejecutar:

```bash
scripts/setup_spinq.sh
```

Este script:

1. Crea un venv 3.9 **separado** del `.venv` principal (3.12):
   `uv venv --python 3.9 .venv-spinq` (`uv` auto-descarga el intérprete
   3.9 si no hay uno instalado en el sistema; no requiere `conda`).
2. Instala `spinqit` **desde PyPI** junto con `numpy` y `qiskit` dentro de
   ese venv: `uv pip install --python .venv-spinq/bin/python spinqit numpy qiskit`.
3. Verifica que `spinqit` se pueda importar (`import spinqit`).

Por qué 3.9: `spinqit` en PyPI solo publica wheels `cp38`/`cp39`/`cp310`
(no 3.11+), por eso va en un entorno aislado del principal 3.12.

Notas:

- `uv pip install <paquete> --python <venv>/bin/python` instala contra el
  intérprete del venv indicado sin tocar el `uv.lock` del proyecto
  principal — `.venv-spinq` es completamente independiente y no tiene su
  propio lockfile administrado por este proyecto.
- `.venv-spinq/` no se commitea (ya está en `.gitignore`).
- Si necesitas ejecutar `main.py` completo (no solo `spinq_backend.py`)
  desde `.venv-spinq`, puede que necesites instalar además el resto de
  dependencias del pipeline clásico (`pandas`, `scikit-learn`,
  `matplotlib`, `scipy`) con la misma herramienta, p.ej.:
  `uv pip install --python .venv-spinq/bin/python pandas scikit-learn matplotlib scipy`.
- `scripts/setup_spinq.sh` también instala `jsonschema` en `.venv-spinq`:
  `persistence/logger.py` (usado para el registro por-circuito de
  ejecuciones en hardware SpinQ) importa `schemas/validators.py`, que a
  su vez depende de `jsonschema`; sin este paquete el import falla en el
  intérprete 3.9 de `.venv-spinq` al correr `main.py` vía
  `scripts/run_spinq.sh`.

## Verificar el guard sin `spinqit` instalado

Desde el entorno principal (3.12) puedes comprobar el mensaje del guard
sin tener `spinqit` disponible:

```bash
uv run python -c "from spinq_backend import check_spinq_environment as c; c()"
```

Como en el entorno principal `spinqit` no está instalado, esto debe
lanzar `SpinQEnvironmentError` explicando que `spinqit` no es importable
y apuntando a este documento. (Una versión de Python distinta de 3.9 por
sí sola **no** provoca el error: solo imprime un aviso.)

## Ejecución: `scripts/run_spinq.sh`

Una vez corrido `scripts/setup_spinq.sh` y configurado `.env` con las
credenciales del NMR, ejecuta:

```bash
scripts/run_spinq.sh
```

Este script verifica que `.venv-spinq` exista y luego corre
`BACKEND_MODE=spinq_nmr .venv-spinq/bin/python main.py`. Gracias a que
`main.py` lee `BACKEND_MODE` desde la variable de entorno del mismo
nombre (con `"spinq_nmr"`/el valor por defecto del archivo como
*fallback*), no hace falta editar `main.py` para cambiar de modo.

Si el entorno 3.9 + `spinqit` no está correctamente configurado,
`main._get_backend()` captura el `SpinQEnvironmentError` del guard,
imprime el mensaje accionable, y hace *fallback* automático a
`statevector` (igual que el resto de backends remotos cuando no están
disponibles) — el pipeline no se interrumpe, pero los resultados
exportados corresponderán a `statevector`, no a hardware SpinQ real.

## Guard de versión/entorno (referencia)

Implementado en `spinq_backend.py`:

- `spinq_backend.check_spinq_environment()`: verifica (sin importar
  `spinqit`) que `importlib.util.find_spec("spinqit")` no sea `None`.
  Ese es el **único requisito duro**: si `spinqit` no es importable,
  lanza `spinq_backend.SpinQEnvironmentError` con un mensaje que apunta a
  los pasos de instalación (resumen de este documento). Adicionalmente,
  si `sys.version_info[:2] != (3, 9)` **solo imprime un aviso** ("probado
  en 3.9 según el tutorial; intentando en Python X.Y") y continúa,
  permitiendo el intento en 3.12.
- `SpinQNMRBackend.connect()` llama a este guard antes de intentar
  `from spinqit import get_nmr, get_compiler`, así el error que ve el
  usuario es accionable en vez de un `ImportError` sin contexto.
- `main._get_backend()` (rama `spinq_nmr`) captura tanto
  `SpinQEnvironmentError` como `ImportError` y hace *fallback* a
  `statevector`, imprimiendo el motivo exacto en cada caso.

## Limitaciones conocidas

- **2 qubits únicamente**: el hardware NMR de SpinQ soporta solo 2 qubits
  (`SpinQNMRBackend._qiskit_to_spinq` asigna `circ.allocateQubits(2)`
  de forma fija). Coincide con `config.N_QUBITS = 2` usado por todo el
  proyecto, pero no generaliza a más qubits.
- **Endianness sin verificar en hardware real** (pendiente desde la Fase
  1): `config.BACKEND_ENDIANNESS["spinq_nmr"] = "big"` es una *suposición*
  basada en cómo `spinq_backend._probabilities_to_counts` construye los
  bitstrings a partir de `[p00, p01, p10, p11]`, no una confirmación
  contra el dispositivo real. Si al correr contra hardware real los
  resultados no tienen sentido (p.ej. accuracy sistemáticamente invertida
  o en torno al azar), revisa primero
  `docs/observable_convention.md` y cambia ese valor a `"little"` — no
  debería requerir ningún otro cambio de código.
- **Mapeo de compuertas HEA/ECR pendiente de Fase 2**: la traducción
  Qiskit → spinqit en `SpinQNMRBackend._qiskit_to_spinq` soporta `ry`,
  `rx`, `rz` y `cx` de forma directa, y decodifica `ecr` de forma
  *aproximada* como `H · CX · H` (ver el comentario en el propio método:
  "ECR(c,t) ≈ (I⊗H) · CNOT · (I⊗H) ... approximate"). Esta aproximación
  no ha sido validada contra la implementación real de ECR en hardware
  IBM ni contra el compilador nativo de spinqit; el ansatz HEA (que usa
  ECR) debe tratarse como no verificado en SpinQ hasta confirmar esa
  equivalencia.
- **Sin acceso al dispositivo ni a `spinqit` en este entorno de
  desarrollo**: nada de lo anterior (conexión real, ejecución de
  circuitos, verificación del guard con `spinqit` genuinamente instalado)
  pudo probarse de punta a punta aquí. Lo que sí se verificó: que
  `spinq_backend.py` se importa sin errores en Python 3.12 sin `spinqit`
  instalado, y que `check_spinq_environment()` lanza el error esperado.
