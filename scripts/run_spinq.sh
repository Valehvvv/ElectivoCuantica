#!/usr/bin/env bash
# Ejecuta main.py en el entorno SpinQ 3.9 con BACKEND_MODE=spinq_nmr.
# Requiere haber corrido scripts/setup_spinq.sh antes y estar en la LAN del NMR.
set -euo pipefail

if [ ! -d ".venv-spinq" ]; then
  echo "[run-spinq] Falta .venv-spinq. Corre primero: scripts/setup_spinq.sh" >&2
  exit 1
fi

BACKEND_MODE=spinq_nmr .venv-spinq/bin/python main.py
