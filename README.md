# Comparación de Ansätze Variacionales de 2 Qubits para Clasificación Binaria

Proyecto universitario de Computación Cuántica. Compara tres arquitecturas
variacionales (Base, Reducido, HEA) para clasificación binaria sobre el
dataset Iris (Versicolor vs. Virginica), ejecutado en múltiples plataformas
cuánticas: simulación local, QRydDemo, SpinQ NMR (universidad) e IBM Quantum.

## Estructura del proyecto

```
electivo/
├── data/                          # Datasets procesados (CSV)
├── notebooks/                     # Notebook original de referencia
├── results/                       # Resultados y gráficos generados
│
├── .env.example                   # Template de variables de entorno
├── config.py                      # Constantes de configuración (carga .env automático)
├── utils.py                       # Utilidades generales (timer, etc.)
├── dataset.py                     # Carga y exportación de datos
├── preprocessing.py               # Pipeline de preprocesamiento
├── circuits.py                    # Definición de los 3 ansätze
├── models.py                      # Clase VQC con soporte multi-backend
├── training.py                    # Entrenamiento con COBYLA + BCE
├── evaluation.py                  # Métricas de clasificación y estructurales
├── visualization.py               # Generación automática de gráficos
├── ibm_runtime.py                 # Integración con IBM Quantum
├── qryd_backend.py                # Integración con QRydDemo (Rydberg)
├── spinq_backend.py               # Integración con SpinQ NMR (universidad)
└── main.py                        # Entry point del pipeline completo
```

## Requisitos

Este proyecto usa [`uv`](https://docs.astral.sh/uv/) para gestionar el
entorno y las dependencias (Python >= 3.12, ver `.python-version`).

```bash
# Instalar uv (si no lo tienes; alternativamente ver la guía oficial de uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar/sincronizar las dependencias del proyecto
uv sync
```

Todos los comandos del proyecto se ejecutan con `uv run ...` (por ejemplo
`uv run python main.py`, `uv run pytest tests/`), lo que asegura que se
usa el entorno virtual gestionado por `uv` (`.venv`).

## Uso

### Ejecución completa (simulación local por defecto)

```bash
uv run python main.py
```

Esto ejecuta el pipeline completo:
1. Carga y preprocesa el dataset Iris (Versicolor vs. Virginica)
2. Aplica LDA + Petal Width, escala a [0, π]
3. Entrena los 3 ansätze (Base, Reducido, HEA) con COBYLA
4. Evalúa métricas de clasificación y estructurales
5. Exporta `results/results.csv` y genera gráficos

### Cambio de backend

Editar únicamente `BACKEND_MODE` en `main.py`:

```python
BACKEND_MODE = "statevector"    # Simulación exacta (default)
BACKEND_MODE = "aer_simulator"  # Qiskit Aer local
BACKEND_MODE = "ibm_simulator"  # Simulador IBM Quantum cloud
BACKEND_MODE = "ibm_hardware"   # Hardware cuántico real IBM
BACKEND_MODE = "spinq_nmr"      # SpinQ NMR 2-qubit (computador U)
BACKEND_MODE = "qryd"           # QRydDemo (emulador de Rydberg)
```

### Configuración para SpinQ NMR (computador cuántico de la universidad)

`spinqit` (el SDK de SpinQ, entregado por el profesor como `.whl`)
requiere Python 3.9 y es incompatible con el entorno principal de este
proyecto (Python >= 3.12). Por eso necesita un **entorno `uv` separado**
(`.venv-spinq`), que **no** se instala con `uv sync`.

Procedimiento completo, reproducible con `uv` (creación del entorno 3.9,
instalación de la `.whl`, variables `.env`, limitaciones conocidas):
**[`docs/spinq_setup.md`](docs/spinq_setup.md)**.

Resumen rápido:

```bash
uv venv --python 3.9 .venv-spinq
source .venv-spinq/bin/activate
uv pip install /ruta/a/spinqit-<version>.whl numpy qiskit
```

Luego configura `.env` (`SPINQ_IP`, `SPINQ_PORT`, `SPINQ_USERNAME`,
`SPINQ_PASSWORD`, `SPINQ_TASK_NAME`), cambia `BACKEND_MODE = "spinq_nmr"`
en `main.py`, y ejecútalo **desde `.venv-spinq` activado**:

```bash
python main.py
```

Si el entorno 3.9 + `spinqit` no está disponible, `main.py` lo detecta
con un mensaje explícito (ver `spinq_backend.check_spinq_environment`) y
hace *fallback* a `statevector` en vez de fallar silenciosamente.

### Configuración para QRydDemo (emulador de átomos de Rydberg)

`qiskit-qryd-provider` requiere Qiskit 1.x (incompatible con Qiskit 2.x del
entorno principal), por lo que necesita un entorno conda separado:

```bash
conda create --name qryd_env python=3.11 -y
conda activate qryd_env
pip install "qiskit>=1.0,<2" qiskit-qryd-provider pandas scikit-learn matplotlib scipy
```

Configurar `.env` con el token de QRydDemo (obtenerlo en
https://theqturer.qryddemo.com):

```env
QRYD_API_TOKEN=tu-token-aqui
QRYD_BACKEND_NAME=qryd_emulator$square
```

Cambiar `BACKEND_MODE = "qryd"` en `main.py` y ejecutar:

```bash
python main.py
```

> **Nota**: QRydDemo tiene una cuota diaria de 5000 unidades. El proyecto
> reduce automáticamente las iteraciones y muestras de entrenamiento cuando
> `BACKEND_MODE = "qryd"` para no superar el límite.

### Configuración para IBM Quantum

1. Copiar el archivo de variables de entorno:

```bash
cp .env.example .env
```

2. Editar `.env` con tu token de IBM Quantum (obtenerlo en https://quantum.ibm.com):

```env
IBMQ_TOKEN=TU-TOKEN-AQUI
IBMQ_INSTANCE=ibm-q/open/main
```

3. `qiskit-ibm-runtime` ya forma parte de las dependencias del proyecto
   (`pyproject.toml`), así que `uv sync` lo instala automáticamente; no
   hace falta instalarlo aparte.

4. Cambiar `BACKEND_MODE` en `main.py`:

```python
BACKEND_MODE = "ibm_simulator"  # Simulador cloud IBM (rápido, sin cola)
BACKEND_MODE = "ibm_hardware"   # Hardware real IBM (ibm_fez por defecto)
```

5. Ejecutar:

```bash
uv run python main.py
```

El archivo `.env` está en `.gitignore` para no exponer tu token. Usa `.env.example` como referencia.

## Ansätze implementados

| Ansatz   | Parámetros | Puertas 2Q | Estructura                              |
|----------|-----------|------------|-----------------------------------------|
| Base     | 6         | 2 CX       | 3 capas Ry + 2 capas de entrelazamiento |
| Reducido | 4         | 1 CX       | 2 capas Ry + 1 capa de entrelazamiento  |
| HEA      | 4         | 1 ECR      | RX/RZ locales + ECR                     |

## Métricas calculadas

**Clasificación:** accuracy, precision, recall, F1-score, ROC AUC, matriz de confusión

**Estructurales:** profundidad, número de compuertas, puertas de 2 qubits

**Temporales:** tiempo de entrenamiento, inferencia y total

## Resultados

Los resultados se guardan automáticamente en `results/results.csv` con columnas:

| Ansatz | Backend | Accuracy | Precision | Recall | F1 | ROC_AUC | Depth | Gate_Count | Two_Qubit_Gates | Training_Time_sec | Inference_Time_sec | Total_Time_sec | Shots | Date |
|--------|---------|----------|-----------|--------|----|---------|-------|------------|-----------------|-------------------|--------------------|----------------|-------|------|

## Notebook original

El notebook `preprocessingfinalfinal.ipynb` se mantiene sin modificar como
referencia del trabajo original.
