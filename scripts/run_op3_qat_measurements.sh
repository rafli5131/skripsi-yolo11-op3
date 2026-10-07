#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "${REPO_ROOT}/../install/setup.bash"
source "${REPO_ROOT}/ros2_ws/install/setup.bash"
PYTHON_BIN="${OP3_PYTHON:-${REPO_ROOT}/../.venv-op3-vision/bin/python}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" scripts/deployment/run_op3_qat_experiments.py "$@"
