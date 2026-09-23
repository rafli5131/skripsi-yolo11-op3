"""Reproducible CPU-only OpenVINO and real-camera benchmark for OP3.

Example:
  python scripts/benchmark_op3.py --model-kind fp32 --model-xml .cache/hardware_models/fp32/model.xml
  python scripts/benchmark_op3.py --mode camera --seconds 60 --model-kind fp32 --model-xml .cache/hardware_models/fp32/model.xml

QAT runs are refused unless scripts/audit_op3_models.py can prove true-QAT
selection, accepted OpenVINO export, and numerical-equivalence evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openvino as ov
import psutil

from op3_model_audit import EXPECTED_CLASSES, qat_validation_gate, sha256_file
from op3_model_audit import ptq_validation_gate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FP32 = ROOT / ".cache/hardware_models/fp32/model.xml"
DEFAULT_PTQ = ROOT / "experiments/ptq/backend_models/ptq_openvino_model/model.xml"
OUTPUT_DIR = ROOT / "experiments/hardware_op3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "camera"), default="synthetic")
    parser.add_argument("--model-kind", choices=("fp32", "ptq_int8", "qat_int8"), required=True)
    parser.add_argument("--model-xml", type=Path, help="Defaults to the audited artifact for model-kind.")
    parser.add_argument("--device", default="CPU", help="OpenVINO device; official OP3 runs must use CPU.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seconds", type=float, default=60.0, help="Camera duration; stability uses 600 seconds.")
    parser.add_argument("--camera-device", help="Audited V4L2 path such as /dev/video0; required in camera mode.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--resource-interval", type=float, default=0.2)
    parser.add_argument("--show", action="store_true", help="Show camera detections for functional verification only.")
    parser.add_argument("--result-stem", help="Override output filename stem; raw CSV and resource CSV use same stem.")
    return parser.parse_args()


def run_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def percentile_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "median", "min", "max", "std", "p50", "p90", "p95", "p99")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array, ddof=0)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_width) / 2.0
    pad_y = (size - new_height) / 2.0
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, scale, left, top


def box_iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, limit: int = 100) -> list[int]:
    if not len(boxes):
        return []
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while order.size and len(kept) < limit:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remainder = order[1:]
        overlaps = box_iou_one(boxes[current], boxes[remainder])
        order = remainder[overlaps <= iou_threshold]
    return kept


def postprocess(
    output: np.ndarray,
    original_shape: tuple[int, int],
    transform: tuple[float, int, int],
    confidence: float,
    iou: float,
) -> list[dict[str, Any]]:
    prediction = np.asarray(output)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise RuntimeError(f"Expected a rank-2 YOLO output after batch removal, got {prediction.shape}.")
    if prediction.shape[0] < prediction.shape[1] and prediction.shape[0] >= 5:
        rows = prediction.T
    elif prediction.shape[1] >= 5:
        rows = prediction
    else:
        raise RuntimeError(f"Could not identify YOLO output orientation: {prediction.shape}.")
    if rows.shape[1] != len(EXPECTED_CLASSES) + 4:
        raise RuntimeError(f"Expected 4 box values + 3 class scores, got {rows.shape[1]} values.")

    class_scores = rows[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(rows)), class_ids]
    selected = np.flatnonzero(scores >= confidence)
    if selected.size > 3000:
        selected = selected[np.argsort(-scores[selected], kind="stable")[:3000]]
    if not selected.size:
        return []

    boxes = rows[selected, :4].astype(np.float32, copy=True)
    boxes[:, 0] -= boxes[:, 2] / 2.0
    boxes[:, 1] -= boxes[:, 3] / 2.0
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    scale, pad_left, pad_top = transform
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
    original_height, original_width = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)

    detections: list[dict[str, Any]] = []
    for class_id in range(len(EXPECTED_CLASSES)):
        class_local = np.flatnonzero(class_ids[selected] == class_id)
        if not class_local.size:
            continue
        kept = nms_indices(boxes[class_local], scores[selected[class_local]], iou)
        for local_index in kept:
            source_index = int(class_local[local_index])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": EXPECTED_CLASSES[str(class_id)],
                    "confidence": float(scores[selected[source_index]]),
                    "x1": float(boxes[source_index, 0]),
                    "y1": float(boxes[source_index, 1]),
                    "x2": float(boxes[source_index, 2]),
                    "y2": float(boxes[source_index, 3]),
                }
            )
    detections.sort(key=lambda item: -item["confidence"])
    return detections[:300]


class ResourceSampler:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        psutil.cpu_percent(interval=None, percpu=True)
        self.process.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._run, name="op3-resource-sampler", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                proc_mem = self.process.memory_info()
                mem = psutil.virtual_memory()
                self.samples.append(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "monotonic_ns": time.perf_counter_ns(),
                        "process_cpu_percent": self.process.cpu_percent(interval=None),
                        "system_cpu_percent": psutil.cpu_percent(interval=None),
                        "per_core_cpu_percent": psutil.cpu_percent(interval=None, percpu=True),
                        "rss_bytes": proc_mem.rss,
                        "vms_bytes": proc_mem.vms,
                        "system_memory_used_bytes": mem.used,
                        "system_memory_available_bytes": mem.available,
                        "system_memory_percent": mem.percent,
                    }
                )
            except (psutil.Error, OSError):
                continue

    def stop(self) -> list[dict[str, Any]]:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=max(1.0, self.interval * 3))
        return self.samples


def resource_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "process_cpu_percent",
        "system_cpu_percent",
        "rss_bytes",
        "vms_bytes",
        "system_memory_used_bytes",
        "system_memory_available_bytes",
        "system_memory_percent",
    )
    result: dict[str, Any] = {"sample_count": len(samples)}
    for field in fields:
        values = [float(item[field]) for item in samples if item.get(field) is not None]
        result[field] = percentile_stats(values)
    if samples and samples[0].get("per_core_cpu_percent"):
        matrix = np.asarray([sample["per_core_cpu_percent"] for sample in samples], dtype=np.float64)
        result["per_core_cpu_percent"] = {
            "mean_by_core": np.mean(matrix, axis=0).tolist(),
            "peak_by_core": np.max(matrix, axis=0).tolist(),
            "p95_by_core": np.percentile(matrix, 95, axis=0).tolist(),
        }
    return result


def save_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    shown = frame.copy()
    colors = ((40, 220, 40), (0, 180, 255), (255, 100, 20))
    for detection in detections:
        class_id = detection["class_id"]
        x1, y1, x2, y2 = (int(round(detection[key])) for key in ("x1", "y1", "x2", "y2"))
        color = colors[class_id % len(colors)]
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.rectangle(shown, (x1, y1), (x2, y2), color, 2)
        cv2.putText(shown, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return shown


def main() -> int:
    args = parse_args()
    if args.device.upper() != "CPU":
        raise SystemExit("Official ROBOTIS OP3 benchmark requires --device CPU.")
    if args.imgsz != 640:
        raise SystemExit("The research input size is fixed at 640; --imgsz must be 640.")
    if args.warmup < 30:
        raise SystemExit("Use at least 30 warmup inferences for a reported benchmark.")
    if args.mode == "synthetic" and args.iterations < 300:
        raise SystemExit("Use at least 300 timed iterations for the standalone benchmark.")
    if args.mode == "camera" and not args.camera_device:
        raise SystemExit("Camera mode requires an audited --camera-device path.")
    if args.model_kind == "qat_int8":
        gate = qat_validation_gate(ROOT)
        if not gate["accepted"]:
            raise SystemExit("BLOCKED: no validated QAT OpenVINO artifact")
    if args.model_kind == "ptq_int8":
        gate = ptq_validation_gate(ROOT, args.model_xml or DEFAULT_PTQ)
        if not gate["accepted"]:
            raise SystemExit("BLOCKED: existing PTQ artifact failed provenance/hash validation")

    default_model = DEFAULT_PTQ if args.model_kind == "ptq_int8" else DEFAULT_FP32
    model_xml = (args.model_xml or default_model).expanduser().resolve()
    model_bin = model_xml.with_suffix(".bin")
    if not model_xml.is_file() or not model_bin.is_file():
        raise SystemExit(f"OpenVINO IR XML/BIN pair not found: {model_xml} and {model_bin}")
    if args.confidence < 0 or args.confidence > 1 or args.iou < 0 or args.iou > 1:
        raise SystemExit("confidence and IoU thresholds must be in [0,1].")

    started = datetime.now(timezone.utc).isoformat()
    compile_start = time.perf_counter_ns()
    core = ov.Core()
    if "CPU" not in core.available_devices:
        raise SystemExit(f"OpenVINO CPU device unavailable; devices={core.available_devices}")
    model = core.read_model(str(model_xml))
    input_shape = model.input(0).get_partial_shape()
    if not input_shape.is_static or list(input_shape.to_shape()) != [1, 3, args.imgsz, args.imgsz]:
        raise SystemExit(f"Unexpected model input shape: {input_shape}")
    input_type = model.input(0).get_element_type().get_type_name()
    if input_type != "f32":
        raise SystemExit(f"Expected FP32 normalized image input, got {input_type}")
    compiled = core.compile_model(model, "CPU")
    request = compiled.create_infer_request()
    compile_ms = (time.perf_counter_ns() - compile_start) / 1e6
    output_shape = list(compiled.output(0).get_partial_shape().to_shape())

    synthetic = np.random.default_rng(42).integers(0, 256, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    capture: cv2.VideoCapture | None = None
    if args.mode == "camera":
        capture = cv2.VideoCapture(args.camera_device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise SystemExit(f"Unable to open audited camera device {args.camera_device}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def one_frame(frame: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        before = time.perf_counter_ns()
        tensor, scale, pad_left, pad_top = letterbox(frame, args.imgsz)
        preprocess_ms = (time.perf_counter_ns() - before) / 1e6
        before = time.perf_counter_ns()
        request.infer({compiled.input(0): tensor})
        output = np.array(request.get_output_tensor(0).data, copy=True)
        inference_ms = (time.perf_counter_ns() - before) / 1e6
        if not np.isfinite(output).all():
            raise RuntimeError("OpenVINO output contains NaN or Inf.")
        before = time.perf_counter_ns()
        detections = postprocess(
            output,
            frame.shape[:2],
            (scale, pad_left, pad_top),
            args.confidence,
            args.iou,
        )
        postprocess_ms = (time.perf_counter_ns() - before) / 1e6
        return {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "finite_output": True,
        }, detections

    try:
        warmup_done = 0
        while warmup_done < args.warmup:
            if capture:
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
            else:
                frame = synthetic
            one_frame(frame)
            warmup_done += 1

        sampler = ResourceSampler(args.resource_interval)
        sampler.start()
        rows: list[dict[str, Any]] = []
        processed = 0
        received = 0
        detection_total = 0
        first_frame_shape = None
        run_start = time.perf_counter_ns()
        run_deadline = time.monotonic() + args.seconds if capture else None
        last_detections: list[dict[str, Any]] = []
        consecutive_capture_failures = 0
        while True:
            if capture and run_deadline is not None and time.monotonic() >= run_deadline:
                break
            if not capture and processed >= args.iterations:
                break
            loop_start = time.perf_counter_ns()
            capture_ms = 0.0
            if capture:
                capture_start = time.perf_counter_ns()
                ok, frame = capture.read()
                capture_ms = (time.perf_counter_ns() - capture_start) / 1e6
                if not ok or frame is None:
                    consecutive_capture_failures += 1
                    if consecutive_capture_failures >= 5:
                        break
                    continue
                consecutive_capture_failures = 0
                received += 1
            else:
                frame = synthetic
            first_frame_shape = first_frame_shape or list(frame.shape)
            measurement, detections = one_frame(frame)
            e2e_stage_ms = capture_ms + measurement["preprocess_ms"] + measurement["inference_ms"] + measurement["postprocess_ms"]
            loop_ms = (time.perf_counter_ns() - loop_start) / 1e6
            processed += 1
            detection_total += len(detections)
            last_detections = detections
            rows.append(
                {
                    "iteration": processed,
                    "capture_ms": capture_ms,
                    "preprocess_ms": measurement["preprocess_ms"],
                    "inference_ms": measurement["inference_ms"],
                    "postprocess_ms": measurement["postprocess_ms"],
                    "end_to_end_stage_sum_ms": e2e_stage_ms,
                    "loop_wall_ms": loop_ms,
                    "detection_count": len(detections),
                    "output_shape": json.dumps(measurement["output_shape"]),
                    "finite_output": measurement["finite_output"],
                }
            )
            if args.show and capture:
                cv2.imshow("YOLO11n OpenVINO CPU", draw_detections(frame, detections))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        elapsed_s = (time.perf_counter_ns() - run_start) / 1e9
        resource_samples = sampler.stop()
    finally:
        if capture:
            capture.release()
        if args.show:
            cv2.destroyAllWindows()

    if processed == 0:
        raise SystemExit("No frames were processed.")
    if args.result_stem:
        stem = args.result_stem
    elif args.mode == "synthetic":
        stem = "benchmark_fp32" if args.model_kind == "fp32" else f"benchmark_{args.model_kind}"
    else:
        stem = "camera_benchmark" if args.model_kind == "fp32" else f"camera_benchmark_{args.model_kind}"
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    if args.mode == "synthetic" and stem == "benchmark_fp32":
        resource_name = "resources_fp32.csv"
    else:
        resource_name = f"resources_{stem}.csv"
    resources_path = OUTPUT_DIR / resource_name
    json_path = OUTPUT_DIR / f"{stem}.json"
    csv_columns = [
        "iteration", "capture_ms", "preprocess_ms", "inference_ms", "postprocess_ms",
        "end_to_end_stage_sum_ms", "loop_wall_ms", "detection_count", "output_shape", "finite_output",
    ]
    resource_columns = [
        "timestamp_utc", "monotonic_ns", "process_cpu_percent", "system_cpu_percent",
        "per_core_cpu_percent", "rss_bytes", "vms_bytes", "system_memory_used_bytes",
        "system_memory_available_bytes", "system_memory_percent",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(csv_path, rows, csv_columns)
    save_csv(resources_path, resource_samples, resource_columns)

    timing_fields = ("capture_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "end_to_end_stage_sum_ms", "loop_wall_ms")
    timing = {field: percentile_stats([float(row[field]) for row in rows]) for field in timing_fields}
    inference_mean = timing["inference_ms"]["mean"]
    wall_fps = processed / elapsed_s if elapsed_s > 0 else None
    resources = resource_summary(resource_samples)
    git_sha = run_command(["git", "rev-parse", "HEAD"])
    git_branch = run_command(["git", "branch", "--show-current"])
    try:
        cpu_description = " ".join(
            line.split(":", 1)[1].strip()
            for line in run_command(["lscpu"]).splitlines()
            if line.strip().startswith("Model name:")
        )
    except AttributeError:
        cpu_description = platform.processor()
    memory = psutil.virtual_memory()
    report = {
        "timestamp_utc": started,
        "git_commit": git_sha,
        "git_branch": git_branch,
        "hostname": platform.node(),
        "cpu_model": cpu_description,
        "system_ram_total_bytes": memory.total,
        "openvino_version": ov.__version__,
        "python_version": platform.python_version(),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "command": " ".join([sys.executable, *sys.argv]),
        "model": {
            "kind": args.model_kind,
            "xml_path": str(model_xml),
            "xml_sha256": sha256_file(model_xml),
            "xml_size_bytes": model_xml.stat().st_size,
            "bin_path": str(model_bin),
            "bin_sha256": sha256_file(model_bin),
            "bin_size_bytes": model_bin.stat().st_size,
        },
        "experimental_setup": {
            "mode": args.mode,
            "device": "CPU",
            "input_resolution": [args.imgsz, args.imgsz],
            "batch": 1,
            "warmups": args.warmup,
            "iterations_requested": args.iterations if args.mode == "synthetic" else None,
            "iterations_processed": processed,
            "duration_requested_seconds": args.seconds if args.mode == "camera" else None,
            "duration_measured_seconds": elapsed_s,
            "camera_device": args.camera_device,
            "camera_frames_received": received if capture is not None else None,
            "camera_frames_processed": processed if capture is not None else None,
            "camera_dropped_frames": "unknown (V4L2/OpenCV capture has no source sequence number)" if capture is not None else None,
            "camera_frame_shape": first_frame_shape,
            "confidence_threshold": args.confidence,
            "iou_threshold": args.iou,
            "visualization_enabled": args.show,
            "resource_sample_interval_seconds": args.resource_interval,
            "compile_time_ms_excluded_from_inference": compile_ms,
        },
        "input": {"shape": [1, 3, args.imgsz, args.imgsz], "dtype": input_type},
        "output": {"shape": output_shape, "dtype": "f32", "finite": all(row["finite_output"] for row in rows)},
        "latency_ms": timing,
        "fps_from_mean_inference_latency": 1000.0 / inference_mean if inference_mean else None,
        "effective_end_to_end_fps": wall_fps,
        "resource_summary": resources,
        "detection_observations": {
            "total_over_processed_frames": detection_total,
            "mean_per_frame": detection_total / processed,
            "last_frame_count": len(last_detections),
            "last_frame": last_detections,
            "accuracy_metrics": "not computed; live frames have no ground truth",
        },
        "raw_measurements_csv": str(csv_path.relative_to(ROOT)),
        "raw_resource_samples_csv": str(resources_path.relative_to(ROOT)),
        "official_minimums_met": args.warmup >= 30 and (args.mode != "synthetic" or args.iterations >= 300),
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)}, {csv_path.relative_to(ROOT)}, {resources_path.relative_to(ROOT)}")
    print(f"mode={args.mode} processed={processed} inference mean={inference_mean:.3f} ms p95={timing['inference_ms']['p95']:.3f} ms")
    print(f"inference FPS={report['fps_from_mean_inference_latency']:.2f}; effective FPS={wall_fps:.2f}")
    print(f"CPU mean={resources['process_cpu_percent']['mean']}; RAM peak RSS={resources['rss_bytes']['max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
