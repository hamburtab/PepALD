#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-molformer_env}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Uni-Dock GPU is officially supported on Linux only."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Uni-Dock GPU requires an NVIDIA driver/runtime."
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH."
  exit 1
fi

echo "Installing Uni-Dock into conda env: ${ENV_NAME}"
conda install -n "${ENV_NAME}" -c conda-forge -y unidock

echo "Installing Uni-Dock Tools (optional Python helpers) into ${ENV_NAME}"
conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}/Uni-Dock/unidock_tools"

echo "Verifying installation"
conda run -n "${ENV_NAME}" unidock --version
conda run -n "${ENV_NAME}" python - <<'PY'
import shutil
print("unidock:", shutil.which("unidock"))
print("unidocktools:", shutil.which("unidocktools"))
PY

echo "Uni-Dock setup complete."
