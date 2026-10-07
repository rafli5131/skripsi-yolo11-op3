#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
OUTPUT_ROOT="${1:-experiments/hardware_op3/comparison_models/$(date -u +%Y%m%dT%H%M%SZ)}"
CAMERA_DEVICE="${OP3_CAMERA_DEVICE:-/dev/video0}"
OV_PYTHON="${OP3_PYTHON:-${REPO_ROOT}/../.venv-op3-vision/bin/python}"
TORCH_PYTHON="${OP3_TORCH_PYTHON:-${REPO_ROOT}/.cache/hardware_models/pytorch_venv/bin/python}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite comparison directory: ${OUTPUT_ROOT}" >&2
  exit 2
fi
test -e "${CAMERA_DEVICE}"
mkdir -p "${OUTPUT_ROOT}"
PYTHONPATH="${REPO_ROOT}/ros2_ws/src/yolo11_op3_vision:${PYTHONPATH:-}" "${OV_PYTHON}" - "${OUTPUT_ROOT}" "${CAMERA_DEVICE}" <<'PY'
import json, sys
from pathlib import Path
from datetime import datetime, timezone
from yolo11_op3_vision.measurement import temperatures_c
temps=temperatures_c()
if temps and max(temps)>=85:
    raise SystemExit('Initial temperature >=85 C; benchmark not started')
protocol={'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'models':['fp32_pytorch','fp32_openvino','ptq_int8','qat_int8'],
          'device':'CPU','batch':1,'input_resolution':[640,640],'warmups':30,'synthetic_iterations':300,
          'camera_seconds':60,'camera_device':sys.argv[2],'camera_resolution':[640,480],
          'confidence_threshold':.25,'iou_threshold':.45,'resource_interval_seconds':.2,
          'visualization':False,'ros_used':False,'synthetic_seed':42,'initial_temperatures_c':temps,
          'identity_policy':'user-supplied artifact paths; no hashes checked',
          'limitations':'sequential runs; camera scenes are not identical; runtime preprocess/postprocess differ'}
(Path(sys.argv[1])/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
PY
for MODE in synthetic camera; do
  EXTRA=()
  if [[ "${MODE}" == camera ]]; then EXTRA=(--camera-device "${CAMERA_DEVICE}" --seconds 60); fi
  echo "Measuring ${MODE}: FP32 PyTorch without OpenVINO"
  "${TORCH_PYTHON}" scripts/deployment/benchmark_op3_pytorch.py --mode "${MODE}" \
    --model artifacts/checkpoints/fp32/best.pt --device cpu --imgsz 640 --warmup 30 --iterations 300 \
    --output-dir "${OUTPUT_ROOT}" --evidence-dir "${OUTPUT_ROOT}/evidence" "${EXTRA[@]}"
  for MODEL in fp32 ptq_int8 qat_int8; do
    echo "Measuring ${MODE}: ${MODEL} OpenVINO"
    "${OV_PYTHON}" scripts/deployment/benchmark_op3.py --mode "${MODE}" --model-kind "${MODEL}" \
      --model-xml "artifacts/openvino/${MODEL}/model.xml" --device CPU --imgsz 640 --warmup 30 --iterations 300 \
      --output-dir "${OUTPUT_ROOT}" --result-stem "${MODE}_${MODEL}" "${EXTRA[@]}"
  done
done
python3 scripts/deployment/build_op3_model_comparison.py --root "${OUTPUT_ROOT}"
