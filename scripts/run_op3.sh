#!/usr/bin/env bash
# One operator entrypoint; selecting a model never starts motion.
set -eo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
MODEL="qat_int8"
ACTION=""
CONDITION="diam"
DRY_RUN=false
CAMERA_DEVICE="${OP3_CAMERA_DEVICE:-/dev/video0}"
OV_PYTHON="${OP3_PYTHON:-${REPO_ROOT}/../.venv-op3-vision/bin/python}"
TORCH_PYTHON="${OP3_TORCH_PYTHON:-${REPO_ROOT}/.cache/hardware_models/pytorch_venv/bin/python}"
ROS_READY=false
WALKING_STARTED=false
usage() {
  echo "Usage: bash scripts/run_op3.sh [--model fp32|qat|ptq|non_openvino]"
  echo "       [--action visual|synthetic|camera|ros|manager|init|start|stop]"
  echo "       [--condition diam|berjalan] [--camera-device /dev/video0] [--dry-run]"
  echo "Without arguments: interactive model and operation menu."
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --action) ACTION="$2"; shift 2;;
    --condition) CONDITION="$2"; shift 2;;
    --camera-device) CAMERA_DEVICE="$2"; shift 2;;
    --dry-run) DRY_RUN=true; shift;;
    --help|-h) usage; exit 0;;
    *) usage >&2; exit 2;;
  esac
done
case "${MODEL}" in qat) MODEL=qat_int8;; ptq) MODEL=ptq_int8;; non_openvino) MODEL=fp32_pytorch;; esac
case "${MODEL}" in fp32|qat_int8|ptq_int8|fp32_pytorch);; *) echo "Unknown model" >&2; exit 2;; esac
case "${CONDITION}" in diam|berjalan);; *) echo "Condition must be diam or berjalan" >&2; exit 2;; esac
run() {
  printf 'Command: '; printf '%q ' "$@"; printf '\n'
  if ! ${DRY_RUN}; then "$@"; fi
}
load_ros() {
  if ! ${DRY_RUN} && ! ${ROS_READY}; then
    source /opt/ros/jazzy/setup.bash || return
    source "${REPO_ROOT}/../install/setup.bash" || return
    source "${REPO_ROOT}/ros2_ws/install/setup.bash" || return
    ROS_READY=true
  fi
}
confirm() {
  if ${DRY_RUN}; then return; fi
  local reply
  read -r -p "$1 Ketik $2: " reply
  [[ "$reply" == "$2" ]] || { echo "Dibatalkan."; return 1; }
}
motion_owner_check() {
  load_ros || return
  if ! ${DRY_RUN}; then
    local nodes
    nodes="$(ros2 node list)" || return
    if [[ "$nodes" == *motion_gateway* ]]; then
      echo "Motion gateway aktif. Gunakan kontrol gateway existing agar ownership gerak tidak bertabrakan." >&2
      return 1
    fi
    if ! pgrep -x op3_manager >/dev/null; then
      echo "op3_manager belum berjalan. Pilih menu Jalankan manager dahulu." >&2
      return 1
    fi
  fi
}
choose_model() {
  echo "1) FP32 OpenVINO"
  echo "2) QAT INT8 OpenVINO"
  echo "3) PTQ INT8 OpenVINO"
  echo "4) FP32 PyTorch tanpa OpenVINO"
  local choice
  read -r -p "Pilih model [1-4]: " choice
  case "$choice" in 1) MODEL=fp32;; 2) MODEL=qat_int8;; 3) MODEL=ptq_int8;; 4) MODEL=fp32_pytorch;; *) echo "Pilihan tidak valid"; return 1;; esac
}
execute() {
  local action="$1" output
  case "$action" in
    visual|synthetic|camera|ros)
      if [[ "$action" != synthetic ]] && ! ${DRY_RUN} && [[ ! -e "$CAMERA_DEVICE" ]]; then
        echo "Camera device tidak ada: $CAMERA_DEVICE" >&2; return 1
      fi
      if [[ "$action" == visual ]] && ! ${DRY_RUN} && [[ -z "${DISPLAY:-}" ]]; then
        echo "Visualisasi membutuhkan DISPLAY. Hubungkan melalui ssh -X." >&2; return 1
      fi
      if [[ "$action" != visual && "$action" != synthetic ]]; then
        confirm "Kondisi aktual robot: ${CONDITION}. Script benchmark hanya mencatat." UKUR || return 1
      fi
      output="${REPO_ROOT}/experiments/hardware_op3/operator_runs/$(date -u +%Y%m%dT%H%M%S)_${MODEL}_${action}_${CONDITION}_$$"
      if [[ "$action" == ros ]]; then
        if [[ "$MODEL" == fp32_pytorch ]]; then
          echo "FP32 tanpa OpenVINO tersedia pada menu kamera/synthetic/visual; menu ROS memakai OpenVINO." >&2; return 1
        fi
        load_ros || return
        run "$OV_PYTHON" scripts/deployment/run_op3_qat_experiments.py \
          --model-kind "$MODEL" --condition "$CONDITION" --operator-confirmed \
          --runs 3 --seconds 60 --camera-device "$CAMERA_DEVICE" --output "$output" || return
      else
        local mode="synthetic" duration=60
        local extra=()
        if [[ "$action" != synthetic ]]; then mode=camera; extra=(--camera-device "$CAMERA_DEVICE"); fi
        if [[ "$action" == visual ]]; then duration="${OP3_VISUAL_SECONDS:-120}"; extra+=(--show); fi
        if [[ "$MODEL" == fp32_pytorch ]]; then
          run "$TORCH_PYTHON" scripts/deployment/benchmark_op3_pytorch.py --mode "$mode" \
            --model artifacts/checkpoints/fp32/best.pt --device cpu --imgsz 640 \
            --warmup 30 --iterations 300 --seconds "$duration" --experiment-condition "$CONDITION" --output-dir "$output" \
            --evidence-dir "$output/evidence" "${extra[@]}" || return
        else
          run "$OV_PYTHON" scripts/deployment/benchmark_op3.py --mode "$mode" --model-kind "$MODEL" \
            --model-xml "artifacts/openvino/${MODEL}/model.xml" --device CPU --imgsz 640 \
            --warmup 30 --iterations 300 --seconds "$duration" --experiment-condition "$CONDITION" --output-dir "$output" "${extra[@]}" || return
        fi
      fi
      if ! ${DRY_RUN}; then echo "Hasil: $output"; fi
      ;;
    manager)
      load_ros || return
      if ! ${DRY_RUN} && pgrep -x op3_manager >/dev/null; then echo "Manager sudah aktif."; return; fi
      confirm "Startup controller hardware dapat mengaktifkan motor. Robot sudah ditopang?" MANAGER || return 1
      if ${DRY_RUN}; then
        run ros2 launch op3_manager op3_manager.launch.py simulation:=false auto_start_init_pose:=false auto_reconnect:=false
      else
        local log_dir="${REPO_ROOT}/experiments/hardware_op3/operator_sessions/$(date -u +%Y%m%dT%H%M%S)_$$"
        mkdir -p "$log_dir"
        nohup ros2 launch op3_manager op3_manager.launch.py simulation:=false auto_start_init_pose:=false auto_reconnect:=false >"$log_dir/manager.log" 2>&1 &
        local manager_pid=$!
        echo "$manager_pid" >"$log_dir/manager_launch.pid"
        echo "Manager launch PID $manager_pid. Log: $log_dir/manager.log"
        echo "Tunggu manager siap sebelum init pose. Manager berjalan terpisah dari menu."
      fi
      ;;
    init)
      motion_owner_check || return 1
      confirm "Robot akan bergerak menuju init pose. Topang robot dan pastikan area bebas." INIT || return 1
      run ros2 topic pub --once --max-wait-time-secs 3 /robotis/base/ini_pose std_msgs/msg/String "{data: 'ini_pose'}" || return
      echo "Tunggu init pose selesai dan robot stabil sebelum start walking."
      ;;
    start)
      motion_owner_check || return 1
      confirm "Walking akan aktif dengan parameter existing. Init pose sudah selesai dan operator siap menghentikan robot?" JALAN || return 1
      run ros2 topic pub --once --max-wait-time-secs 3 /robotis/enable_ctrl_module std_msgs/msg/String "{data: 'walking_module'}" || return
      run ros2 topic pub --once --max-wait-time-secs 3 /robotis/walking/command std_msgs/msg/String "{data: 'start'}" || return
      WALKING_STARTED=true
      CONDITION=berjalan
      ;;
    stop)
      load_ros || return
      run ros2 topic pub --once --max-wait-time-secs 3 /robotis/walking/command std_msgs/msg/String "{data: 'stop'}" || return
      WALKING_STARTED=false
      CONDITION=diam
      if ${DRY_RUN}; then echo "Dry-run: perintah stop disiapkan."; else echo "Stop walking dikirim; ini bukan pemutusan daya motor."; fi
      ;;
    *) echo "Unknown action: $action" >&2; return 2;;
  esac
}
cleanup() {
  if ${WALKING_STARTED} && ! ${DRY_RUN}; then
    ros2 topic pub --once --max-wait-time-secs 3 /robotis/walking/command std_msgs/msg/String "{data: 'stop'}" || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ -n "$ACTION" ]]; then
  execute "$ACTION"
  if [[ "$ACTION" == start ]]; then
    # CLI start intentionally persists until the operator issues CLI stop.
    WALKING_STARTED=false
  fi
  exit
fi
choose_model
while true; do
  echo
  echo "Model: ${MODEL} | Camera: ${CAMERA_DEVICE}"
  echo "1) Visual kamera (120 detik; SSH -X; q/Esc untuk selesai)"
  echo "2) Benchmark CPU synthetic (300 iteration)"
  echo "3) Benchmark kamera standalone (60 detik)"
  echo "4) Benchmark ROS OpenVINO (3 × 60 detik)"
  echo "5) Jalankan manager hardware tanpa auto init pose"
  echo "6) Init pose"
  echo "7) Start walking"
  echo "8) Stop walking"
  echo "9) Pilih model lain"
  echo "0) Keluar (stop walking yang dimulai dari menu ini)"
  read -r -p "Pilih operasi: " operation || break
  case "$operation" in
    0) break;;
    9) choose_model || true; continue;;
    1) action=visual;; 2) action=synthetic;; 3) action=camera;; 4) action=ros;;
    5) action=manager;; 6) action=init;; 7) action=start;; 8) action=stop;;
    *) echo "Pilihan tidak valid"; continue;;
  esac
  if [[ "$action" == camera || "$action" == ros ]]; then
    read -r -p "Kondisi aktual [diam/berjalan]: " CONDITION
    case "$CONDITION" in diam|berjalan);; *) echo "Kondisi tidak valid"; continue;; esac
  fi
  execute "$action" || echo "Operasi tidak selesai; periksa pesan di atas."
done
