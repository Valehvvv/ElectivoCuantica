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

- **`feature/IBM-architecture`** — Integración arquitectura IBM Quantum con
  stack de observabilidad completo: logger, resiliencia, snapshot/recuperación
  de estado. Incluye guardrails de costo para QPU real (límite de tiempo +
  logging de uso) y auto-selección de la QPU IBM menos ocupada. Agrega modo
  dry-run `IBM_FAKE` para testing sin consumir cuota. 126 archivos,
  +10321 líneas vs `main`.

- **`feature/refactor-vqc-ibm`** — Refactor del VQC que agrega soporte
  QRydDemo (simulador de átomos de Rydberg) y corrige endianness de SpinQ NMR
  (big-endian) más soporte de `N_TEST`. Comparte infraestructura de
  persistencia/resiliencia con la rama IBM. 127 archivos, +10477 líneas vs
  `main`.

Ambas ramas feature construyen infraestructura casi idéntica (abstracción de
backends, persistencia, resiliencia, schemas, tests) pero divergen en
integraciones específicas: `feature/IBM-architecture` usa `ibm_runtime.py` +
Qiskit Runtime SamplerV2; `feature/refactor-vqc-ibm` usa `qryd_backend.py`
(Rydberg) + fix de endianness SpinQ NMR.

## Notebook original

El notebook `preprocessingfinalfinal.ipynb` se mantiene sin modificar como
referencia del trabajo original.
