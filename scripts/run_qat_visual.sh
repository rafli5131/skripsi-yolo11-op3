#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "ERROR: DISPLAY is empty. Reconnect with SSH X forwarding, for example: ssh -X user@robot." >&2
  exit 2
fi

if [[ -n "${OP3_PYTHON:-}" ]]; then
  PYTHON_BIN="${OP3_PYTHON}"
elif [[ -x "${REPO_ROOT}/../.venv-op3-vision/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/../.venv-op3-vision/bin/python"
else
  PYTHON_BIN="python3"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
export PYTHONUNBUFFERED=1
CAMERA_DEVICE="${OP3_CAMERA_DEVICE:-/dev/video0}"

if [[ ! -e "${CAMERA_DEVICE}" ]]; then
  echo "ERROR: audited camera device does not exist: ${CAMERA_DEVICE}" >&2
  echo "Set OP3_CAMERA_DEVICE to the audited camera path, for example:" >&2
  echo "  OP3_CAMERA_DEVICE=/dev/video1 bash scripts/run_qat_visual.sh" >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" scripts/deployment/benchmark_op3.py \
  --mode camera \
  --model-kind qat_int8 \
  --camera-device "${CAMERA_DEVICE}" \
  --show \
  "$@"
