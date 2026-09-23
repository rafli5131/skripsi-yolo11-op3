# OP3 Hardware Test Report

## 1. Hardware

Measurements were run on the currently connected host `humanoid-NUC12WSHi5`, which reports a 12th Gen Intel Core i5-1240P, 12 physical cores / 16 logical threads, and 7.33 GiB RAM. It exposes AVX, AVX2, and AVX-VNNI; AVX-512 is unavailable. Ubuntu 24.04.5 LTS and kernel 7.0.0-31-generic were detected. The configured target in the thesis is an Intel NUC11PAHi5; the connected host is a NUC12WSHi5, so these figures do **not** certify performance on the specified NUC11 configuration.

The attached Logitech HD Pro Webcam C920 was identified by `v4l2-ctl` and `lsusb`. `/dev/video0` returned camera frames (640 x 480); `/dev/video1` was also enumerated but was not selected. Thermal-zone readings were available without `sudo`; `sensors` was not installed.

## 2. Software

| Component | Observed |
| --- | --- |
| OS | Ubuntu 24.04.5 LTS |
| ROS 2 | Jazzy (`ROS_DISTRO=jazzy`; topics/nodes callable) |
| Python runtime | 3.12.3 in `/home/humanoid/ros2_jazzy/.venv-op3-vision` |
| OpenVINO runtime | 2026.3.1; available device `CPU` |
| OpenCV / camera | V4L2 capture succeeded on `/dev/video0` |
| Inference target | OpenVINO CPU, batch 1, 640 x 640 |

The repository's pinned runtime is OpenVINO 2024.4 and its project metadata requires Python below 3.12. The measurements use the already-installed ROS workspace virtual environment instead (Python 3.12.3 / OpenVINO 2026.3.1); repeat with the pinned stack before treating these measurements as directly comparable to a pinned-runtime experiment. No NVIDIA device was selected or used.

## 3. Model provenance

The tested deployment model was the repository's **YOLO11n FP32 OpenVINO IR**. The checkpoint is `artifacts/checkpoints/fp32/best.pt` (5,481,875 bytes; SHA256 `e05b2b21f7faa68f65671f0aaae37ad962aa5e25c6b8b26853fd0b389d420b8c`). Its manifest says `completed`, its class mapping is `0: ball`, `1: gawang`, `2: robot`, and the FP32 export provenance agrees with the repository's validation manifest. XML and BIN were hydrated from Git LFS into the ignored `.cache` directory; both runtime files match their Git LFS object IDs:

| File | Size | SHA256 |
| --- | ---: | --- |
| `model.xml` | 324,944 bytes | `233bb872e218ab3bf525188b0e105eea9f2a79e60ffecb4b767e63939ba257e7` |
| `model.bin` | 10,431,992 bytes | `e85f45438b5f7c8858b1a168a03f61b12a73b8159391529d19b84e09ff854026` |

The true-QAT checkpoint `artifacts/checkpoints/qat/qat_best.pt` is confirmed as NNCF QAT from its manifest (`create_compressed_model`), checkpoint hash, and matching LFS OID (SHA256 `f96911a1b3f4f9a8987a7fb57611aac2455b2b5c7b93c27db146d69d5013a8c8`). It was not deployed as OpenVINO. The required selection manifest is absent, and the available QAT export/equivalence diagnostics are rejected, unresolved, or stopped. The historical `artifacts/openvino/qat_int8/model.xml` is therefore not loaded or benchmarked.

**BLOCKED: no validated QAT OpenVINO artifact.** The QAT INT8 artifact cannot be called valid because there is no accepted selection evidence and no accepted numerical-equivalence report for the export. No QAT OpenVINO hardware result is reported. No training, fine-tuning, PTQ, or calibration was performed.

## 4. Experimental setup

Standalone tests used CPU, batch 1, 640 x 640, 30 warmup inferences, and 300 measured inferences for the synthetic run. Model load and compilation are excluded from latency. Camera and ROS tests ran with visualization/debug-image publishing disabled. The standalone C920 test lasted 60 seconds. ROS benchmarking discarded 30 warmup callbacks; the stability test ran for 600.03 seconds after those warmups. Timing uses `time.perf_counter_ns()` and raw event/resource samples are retained as CSV.

The benchmark scripts record the checked-out HEAD at measurement time (`c5fe60711f615e71aa73f54586d8b16ffac96122`), hostname, CPU, RAM, Python/OpenVINO/ROS versions, model SHA256, command, device, resolution, batch, warmup count, and iteration/duration. The benchmark scripts were uncommitted while measurements ran; their measured source is now committed in the thesis repository at `7aab807631732a263c12db802759fdda11b9d130`, and the ROS node/launch integration is committed in the workspace repository at `d13a18e3ef661f6600663236deb8bbd94fa6f024`. Result JSON distinguishes the measurement-time HEAD from those source commits and marks the worktree dirty. Process CPU is psutil's aggregate across cores (100% equals one logical core), so values above 100% indicate multicore use. Resource sampling interval was 0.2 seconds. The host also had other ROS workspace activity; system CPU is provided to show concurrent system load.

## 5. Standalone benchmark

The 300-iteration synthetic run compiled the model on CPU and successfully inferred finite output with input `[1, 3, 640, 640]` (`f32`) and output `[1, 7, 8400]` (`f32`). The model's mean inference latency was 28.491 ms (P95 30.350 ms), corresponding to 35.10 inference FPS. Including the measured preprocessing and postprocessing stage sum, mean latency was 30.433 ms and effective loop throughput was 32.77 FPS.

| Measurement | Inference latency mean / P95 | Effective FPS | Process CPU mean / P95 | Process RSS mean / peak |
| --- | ---: | ---: | ---: | ---: |
| Synthetic, 300 inferences | 28.491 / 30.350 ms | 32.77 | 873.5% / 901.3% | 227.5 / 227.6 MiB |

Raw timings and 0.2-second resource samples: `benchmark_fp32.csv` and `resources_fp32.csv`; metadata and summaries: `benchmark_fp32.json`.

## 6. Camera benchmark

The standalone pipeline captured and processed 1,273 frames in 60.04 seconds. The camera source did not expose sequence numbers through this OpenCV/V4L2 path, so camera-side dropped frames cannot be counted reliably. Mean stage times were capture 7.837 ms, preprocessing 1.770 ms, inference 37.067 ms, and postprocessing 0.393 ms. The sum of those stages averaged 47.067 ms; observed effective throughput was 21.20 frames/s. Inference-only throughput was 26.98 FPS. The final observed frame contained a `ball` detection at confidence 0.938; this is a functional observation, not an accuracy measurement.

| Measurement | Inference latency mean / P95 | Effective FPS | Process CPU mean / P95 | Process RSS mean / peak |
| --- | ---: | ---: | ---: | ---: |
| C920 standalone, 60 s | 37.067 / 60.078 ms | 21.20 | 716.2% / 818.2% | 229.2 / 229.8 MiB |

Raw timings and resource samples: `camera_benchmark.csv` and `resources_camera_benchmark.csv`; metadata: `camera_benchmark.json`.

## 7. ROS 2 benchmark

The existing `op3_advanced_detector` package was extended with a CPU-only YOLO11 OpenVINO node and launch files. The node subscribes to `/image_raw`, publishes `vision_msgs/Detection2DArray` on `/vision/detections`, and publishes timing telemetry on `/vision/benchmark`. Each detection carries its numeric ID and canonical class name as `class_id` (`0:ball`, `1:gawang`, `2:robot`) plus confidence and bounding box. Optional `/vision/debug_image` publishing is controlled by `publish_debug_image` and defaults to false. Parameters include model path, confidence threshold, IoU threshold, input topic, device, image size, and debug publishing.

The ROS package integration is committed locally in the workspace repository at `d13a18e3ef661f6600663236deb8bbd94fa6f024`; unrelated pre-existing workspace edits were left unstaged. The thesis repository commit contains the audit, benchmark, comparison, and report artifacts.

The package built into the workspace install and the C920 launch was exercised. A 60.03-second run processed 1,454 measured frames with zero exceptions. Mean inference latency was 31.234 ms (P95 54.827 ms); callback latency was 33.828 ms (P95 59.505 ms); effective end-to-end throughput was 24.22 FPS. The node published detections on the requested topic. The ROS benchmark estimated 347 dropped frames against the camera's configured 30 FPS; this is an estimate based on configured source FPS and callback count, not a camera sequence-number count.

| Measurement | Inference latency mean / P95 | Callback latency mean / P95 | Effective FPS | Process CPU mean / P95 | Process RSS mean / peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| C920 ROS 2, 60 s | 31.234 / 54.827 ms | 33.828 / 59.505 ms | 24.22 | 705.1% / 863.9% | 262.2 / 262.2 MiB |

Raw timings and resource samples: `ros2_benchmark.csv` and `resources_ros2_benchmark.csv`; metadata: `ros2_benchmark.json`.

## 8. CPU utilization

During standalone synthetic inference, process CPU averaged 873.5% (P95 901.3%). During direct camera inference it averaged 716.2% (P95 818.2%). The 60-second ROS run averaged 705.1% process CPU (P95 863.9%); system CPU averaged 46.8%. The stability run averaged 705.4% process CPU (P95 857.3%); system CPU averaged 47.2%. These process percentages are aggregate multicore use, not a percentage of the whole 16-thread host. The ROS process could consume roughly seven logical cores on average in the measured camera workloads.

## 9. RAM utilization

The synthetic benchmark process used a mean 227.5 MiB RSS and a 227.6 MiB peak. The direct camera run used 229.2 MiB mean and 229.8 MiB peak. ROS averaged 262.2 MiB, peaking at 262.2 MiB during the 60-second run and 262.4 MiB during the stability run. System memory use averaged 62.3% during stability. RSS remained effectively flat during the 10-minute test; the measured first-to-last sample increase was 28,672 bytes (0.0104%).

## 10. Stability

The camera-backed ROS node completed 600.03 seconds with 14,455 successfully processed measured frames, 30 warmup callbacks, and zero callback exceptions. Effective throughput was 24.09 FPS. Mean inference latency was 31.497 ms (P95 55.022 ms), and mean callback latency was 34.213 ms (P95 59.226 ms). Peak sampled thermal-zone temperature was 76 °C, below the 90 °C abort threshold. The estimated camera frame shortfall was 3,546 against configured 30 FPS; it is an estimate rather than a sequence-number count. RSS increased by 28.7 kB from first to last sample, with no material memory growth observed during this run.

Raw callbacks and resource samples: `stability_test.csv` and `resources_stability_test.csv`; metadata: `stability_test.json`.

## 11. Limitations

- The connected host reports NUC12WSHi5 / i5-1240P, not the specified NUC11PAHi5. Hardware numbers must be re-measured on the intended NUC11 before claiming OP3 NUC11 performance.
- Runtime Python/OpenVINO versions differ from the repository pins, as described above.
- Direct camera dropped-frame counts are unavailable through the selected V4L2/OpenCV capture interface; ROS dropped-frame figures are estimates based on the configured 30 FPS.
- Concurrent processes in the ROS workspace were active during measurements. The system CPU and process CPU records are retained to contextualize load.
- Camera observations have no labels and were not used for precision, recall, F1, or mAP.
- QAT OpenVINO remains blocked by missing checkpoint-selection acceptance and missing accepted numerical-equivalence evidence. Only the valid FP32 OpenVINO artifact was used for deployment benchmarks.
- `ros2 --version` is not supported by the installed Jazzy CLI. ROS distro, nodes, topics, imports, launch, and publication were verified independently.

## 12. Final results

Dataset accuracy comes only from the repository's existing held-out TEST evaluation (`experiments/final_test/`). These values are separate from live hardware observations:

| Model/evaluation artifact | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11n FP32, TEST | 0.93135 | 0.95057 | 0.94086 | 0.96423 | 0.77991 |
| YOLO11n true QAT native PyTorch, TEST | 0.85415 | 0.93752 | 0.89390 | 0.95119 | 0.65670 |

Hardware performance for the valid deployed FP32 OpenVINO model:

| Context | Inference latency mean / P95 | End-to-end FPS | Process CPU mean | Process RSS peak |
| --- | ---: | ---: | ---: | ---: |
| Synthetic standalone | 28.491 / 30.350 ms | 32.77 | 873.5% | 227.6 MiB |
| C920 standalone | 37.067 / 60.078 ms | 21.20 | 716.2% | 229.8 MiB |
| C920 through ROS 2 (60 s) | 31.234 / 54.827 ms | 24.22 | 705.1% | 262.2 MiB |
| C920 through ROS 2 (600 s) | 31.497 / 55.022 ms | 24.09 | 705.4% | 262.4 MiB |

No QAT OpenVINO latency, FPS, CPU, or RAM value is available or inferred. The artifact status remains **BLOCKED — QAT OpenVINO artifact has not passed export/equivalence validation**.

## 13. Reproduction commands

Run from the thesis repository root. First hydrate the validated FP32 IR (Git LFS is preferred; direct media URLs are included for this host because the Git LFS client was unavailable):

```bash
mkdir -p .cache/hardware_models/fp32
curl -fL https://media.githubusercontent.com/media/rafli5131/skripsi-yolo11-op3/main/artifacts/openvino/fp32/model.xml -o .cache/hardware_models/fp32/model.xml
curl -fL https://media.githubusercontent.com/media/rafli5131/skripsi-yolo11-op3/main/artifacts/openvino/fp32/model.bin -o .cache/hardware_models/fp32/model.bin
sha256sum .cache/hardware_models/fp32/model.xml .cache/hardware_models/fp32/model.bin
```

The two hashes must match the values in section 3. Activate the same runtime and workspace overlays used for the ROS test:

```bash
source /opt/ros/jazzy/setup.bash
source /home/humanoid/ros2_jazzy/install/setup.bash
source /home/humanoid/ros2_jazzy/.venv-op3-vision/bin/activate
```

Synthetic standalone benchmark (30 warmups, 300 measured iterations):

```bash
python scripts/benchmark_op3.py --model-kind fp32 --model-xml .cache/hardware_models/fp32/model.xml --device CPU --warmup 30 --iterations 300
```

Standalone camera benchmark (visualization remains off):

```bash
python scripts/benchmark_op3.py --mode camera --seconds 60 --camera-device /dev/video0 --model-kind fp32 --model-xml .cache/hardware_models/fp32/model.xml --device CPU
```

In a separate terminal, start ROS camera and detector, then measure the node from the first terminal:

```bash
ros2 launch op3_advanced_detector yolo11_openvino_camera.launch.py model_path:=/home/humanoid/ros2_jazzy/skripsi-yolo11-op3/.cache/hardware_models/fp32/model.xml video_device:=/dev/video0 input_topic:=/image_raw python_prefix:=/home/humanoid/ros2_jazzy/.venv-op3-vision/bin/python publish_debug_image:=false
python scripts/benchmark_op3_ros2.py --model-kind fp32 --model-xml .cache/hardware_models/fp32/model.xml --seconds 60 --warmup 30 --camera-fps 30
```

For the 10-minute stability run, use the same launch and replace the monitor command with:

```bash
python scripts/benchmark_op3_ros2.py --model-kind fp32 --model-xml .cache/hardware_models/fp32/model.xml --seconds 600 --warmup 30 --camera-fps 30 --result-stem stability_test
```

Hardware audit and model provenance gate:

```bash
python scripts/audit_hardware_op3.py
python scripts/audit_op3_models.py
```

The QAT benchmark runner intentionally exits with `BLOCKED: no validated QAT OpenVINO artifact` until the repository contains accepted selected-checkpoint provenance and accepted numerical-equivalence evidence.
