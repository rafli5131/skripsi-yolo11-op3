# Isolated ROS 2 workspace for YOLO11 OP3 thesis inference

This workspace contains a new `yolo11_op3_vision` package and does not add or modify packages in the robot's main `src/` tree. It subscribes to a `sensor_msgs/Image`, performs OpenVINO CPU inference at 640×640, and publishes `vision_msgs/Detection2DArray` on `/vision/detections`. Each hypothesis uses `class_id` in the form `0:ball`, `1:gawang`, or `2:robot`; bounding boxes are in pixel coordinates through `bbox.center` and `bbox.size_x/size_y`.

The node parameters include `model_path`, `confidence_threshold`, `iou_threshold`, `input_topic`, `device`, `imgsz`, and `publish_debug_image`. Debug publication defaults to `false`. A timed ROS benchmark can set `benchmark_seconds` and `output_dir`; it writes callback latency/FPS, CPU/RAM summaries, and raw timing/resource CSV files. `evidence_image_path` optionally saves one annotated frame for functional verification.

Build only this isolated workspace:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

Use a Python runtime that has both ROS 2 Jazzy Python modules and OpenVINO installed. Choose `model_path` from the canonical `artifacts/openvino/{fp32,ptq_int8,qat_int8}/model.xml` locations after the Git LFS files are available. The node rejects unresolved LFS pointer files and requires the paired BIN.

Example functional run, with an image publisher already publishing `/image_raw`:

```bash
ros2 launch yolo11_op3_vision detector.launch.py \
  model_path:=/path/to/skripsi-yolo11-op3/artifacts/openvino/fp32/model.xml \
  input_topic:=/image_raw \
  python_prefix:=/path/to/openvino-enabled-venv/bin/python
```

Example 60-second ROS benchmark with an optional evidence frame:

```bash
ros2 launch yolo11_op3_vision detector.launch.py \
  model_path:=/path/to/skripsi-yolo11-op3/artifacts/openvino/fp32/model.xml \
  input_topic:=/image_raw benchmark_seconds:=60 \
  python_prefix:=/path/to/openvino-enabled-venv/bin/python \
  output_dir:=/path/to/skripsi-yolo11-op3/experiments/hardware_op3 \
  result_stem:=ros2_benchmark_fp32_openvino \
  evidence_image_path:=/path/to/skripsi-yolo11-op3/experiments/hardware_op3/evidence/live_detection_ros2_fp32.png
```

## Final measurement runner

From the repository root:

```bash
bash scripts/run_op3_qat_measurements.sh --condition diam --operator-confirmed --runs 3 --seconds 60 --camera-device /dev/video0
python3 scripts/deployment/build_op3_thesis_tables.py --root experiments/hardware_op3/final_qat
```

For later walking measurements replace `diam` with `berjalan` only after the operator starts walking and confirms the actual condition. The runner sends no motion commands. Camera device must be audited first; protocol is frozen across conditions and existing raw data is never overwritten.

New parameters: `warmup_camera_frames` (30), `experiment_condition`, `run_id`, `final_experiment`, `publish_benchmark` (false), `benchmark_topic`, and `thermal_limit_c` (90). Model warmup is separate from camera warmup. Primary latency stops after postprocess, before ROS publishing. CSV records stream to disk while measuring; summaries use sample SD (ddof=1). System CPU/RAM are the primary resource metrics; process CPU/RSS are also retained. MB means bytes/1,000,000. Debug and evidence rendering are OFF for official runs.
