# SpinQ NMR: entorno dedicado (Fase 5)

## Por qué un entorno separado

El proyecto principal (`pyproject.toml`, `uv.lock`) usa **Python >= 3.12**
y `qiskit >= 2.5.0` / `numpy >= 2.5.0`. El SDK de SpinQ (`spinqit`,
entregado por el profesor como un archivo `.whl`, **no publicado en
PyPI**) está **documentado contra Python 3.9** en el tutorial del profesor
(la versión recomendada), pero **no es un requisito duro**: puede que
funcione también en 3.12. Un venv 3.9 dedicado es una **opción de
respaldo** por si el intento en 3.12 falla; en ese caso `spinqit` suele
traer además versiones de `numpy`/`qiskit` mucho más antiguas que no
conviven con las del proyecto principal ni en el mismo `uv.lock`.

**Primero intenta `spinqit` en el entorno principal 3.12.** Solo si ese
intento falla, crea el venv 3.9 descrito más abajo.

**Solo el modo `BACKEND_MODE = "spinq_nmr"` necesita este entorno 3.9.**
Todo lo demás del proyecto (`statevector`, `aer_simulator`, `ibm_simulator`,
`ibm_hardware`, la suite de tests, el pipeline clásico) sigue corriendo
normalmente en el entorno principal (Python 3.12, `uv sync` / `uv run`)
sin ningún cambio.

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
  `spinqit` no está en PyPI. El entorno 3.9 real se crea aparte, como se
  describe abajo, sin tocar el `uv.lock` del proyecto principal.
- `spinq_backend.py` sigue haciendo el `import spinqit` de forma
  **perezosa** (dentro de `connect()`), tal como antes de la Fase 5, de
  modo que **importar el módulo en Python 3.12 nunca falla** aunque
  `spinqit` no esté instalado.
- Se añadió un *guard* de entorno (`spinq_backend.check_spinq_environment`,
  ver más abajo) que se ejecuta antes de intentar `import spinqit`, para
  dar un mensaje accionable en vez de un `ImportError` genérico o un
  fallo silencioso.

## Crear el entorno Python 3.9 con `uv` (respaldo)

Este paso solo es necesario **si el intento en 3.12 falla** (por ejemplo,
`spinqit` no importa o revienta en runtime por incompatibilidad de
versión). 3.9 es la versión del tutorial del profesor, por lo que es la
más segura como respaldo.

Desde la raíz del proyecto (`ElectivoCuantica/`):

```bash
# 1. Crear un venv 3.9 SEPARADO del .venv principal (3.12)
uv venv --python 3.9 .venv-spinq

# 2. Activarlo
source .venv-spinq/bin/activate      # bash/zsh
# .venv-spinq\Scripts\activate       # Windows

# 3. Instalar spinqit desde el .whl del profesor (ruta local, NO PyPI)
uv pip install /ruta/a/spinqit-<version>-py3-none-any.whl

# 4. Instalar las dependencias mínimas que necesita spinq_backend.py
#    para traducir circuitos Qiskit -> spinqit. Ajusta las versiones a
#    las que exija spinqit en su propio requirements/README (suelen ser
#    versiones antiguas, p.ej. numpy 1.21.x); si spinqit no fija una
#    versión de qiskit, usa la última compatible con Python 3.9.
uv pip install numpy qiskit

# (Opcional) si vas a ejecutar main.py completo desde este entorno,
# instala también el resto de dependencias del pipeline clásico que uses
# (pandas, scikit-learn, matplotlib, scipy) con la misma herramienta:
uv pip install pandas scikit-learn matplotlib scipy
```

Notas:

- `uv venv --python 3.9` descarga automáticamente un intérprete 3.9
  gestionado por `uv` si no hay uno instalado en el sistema (no requiere
  `conda`).
- `uv pip install <ruta-local>.whl` instala la wheel local directamente,
  sin intentar resolverla contra PyPI ni añadirla al `uv.lock` del
  proyecto principal — el entorno `.venv-spinq` es completamente
  independiente y no tiene su propio lockfile administrado por este
  proyecto.
- `.venv-spinq/` no debe commitearse (añádelo a `.gitignore` si el
  repositorio pasa a usar control de versiones).

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

## Ejecutar `main.py` en modo `spinq_nmr`

1. Copia `.env.example` a `.env` (si no existe) y completa las variables:

```env
SPINQ_IP=IP_DEL_COMPUTADOR
SPINQ_PORT=8989
SPINQ_USERNAME=USUARIO_SPINQ
SPINQ_PASSWORD=CONTRASENA_SPINQ
SPINQ_TASK_NAME=VQC-Experiment
```

2. Edita `main.py` y cambia:

```python
BACKEND_MODE: str = "spinq_nmr"
```

3. Ejecuta `main.py` **desde el entorno `.venv-spinq` activado** (no con
   `uv run`, que usaría el `.venv` principal 3.12):

```bash
source .venv-spinq/bin/activate
python main.py
```

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
