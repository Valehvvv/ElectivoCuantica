# Comparación de Ansätze Variacionales de 2 Qubits para Clasificación Binaria

Proyecto universitario de Computación Cuántica. Compara tres arquitecturas
variacionales (Base, Reducido, HEA) para clasificación binaria sobre el
dataset Iris (Versicolor vs. Virginica), preparado para ejecución tanto en
simulación local como en hardware cuántico real de IBM Quantum.

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
└── main.py                        # Entry point del pipeline completo
```

## Requisitos

```bash
pip install numpy pandas matplotlib scikit-learn qiskit scipy
# Opcional para Aer:
pip install qiskit-aer
# Opcional para IBM Quantum:
pip install qiskit-ibm-runtime
```

## Uso

### Ejecución completa (simulación local por defecto)

```bash
python main.py
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
```

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

3. Instalar el runtime de IBM:

```bash
pip install qiskit-ibm-runtime
```

4. Cambiar `BACKEND_MODE` en `main.py`:

```python
BACKEND_MODE = "ibm_simulator"  # Simulador cloud IBM (rápido, sin cola)
BACKEND_MODE = "ibm_hardware"   # Hardware real IBM (ibm_fez por defecto)
```

5. Ejecutar:

```bash
python main.py
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
