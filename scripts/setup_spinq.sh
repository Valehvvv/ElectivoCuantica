#!/usr/bin/env bash
# Setup entorno aislado SpinQ NMR (Python 3.9), separado del env principal 3.12.
# spinqit (PyPI) solo tiene wheels cp38/cp39/cp310, por eso venv dedicado.
set -euo pipefail

VENV=".venv-spinq"
echo "[setup-spinq] Creando venv Python 3.9 en $VENV ..."
uv venv --python 3.9 "$VENV"          # uv auto-descarga CPython 3.9 si falta

echo "[setup-spinq] Instalando spinqit + runtime stack (numpy<2 pandas scikit-learn matplotlib scipy qiskit jsonschema) ... (numpy<2 por compat autograd/spinqit)"
uv pip install --python "$VENV/bin/python" spinqit "numpy<2" pandas scikit-learn matplotlib scipy qiskit jsonschema

echo "[setup-spinq] Verificando import ..."
"$VENV/bin/python" -c "import spinqit; print('spinqit OK')"

echo "[setup-spinq] Listo."
echo "  Configura IP y credenciales en .env (SPINQ_IP, SPINQ_USERNAME, SPINQ_PASSWORD)."
echo "  Ejecuta en la maquina cuantica: scripts/run_spinq.sh"
