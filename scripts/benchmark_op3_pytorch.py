"""CPU-only native PyTorch FP32 and camera benchmark for ROBOTIS OP3.

This deliberately benchmarks the repository's unchanged FP32 checkpoint through
Ultralytics/PyTorch, without exporting or compiling an OpenVINO model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
import psutil
import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / ".cache/hardware_models/fp32_pytorch/best.pt"
MODEL_MANIFEST = ROOT / "artifacts/checkpoints/fp32/manifest.json"
OUTPUT_DIR = ROOT / "experiments/hardware_op3"
EXPECTED_NAMES = {0: "ball", 1: "gawang", 2: "robot"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "camera"), default="synthetic")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--camera-device")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--resource-interval", type=float, default=0.2)
    parser.add_argument("--evidence-dir", type=Path, default=OUTPUT_DIR / "evidence")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_expected_checkpoint_hash() -> str:
    pointer_path = ROOT / "artifacts/checkpoints/fp32/best.pt"
    try:
        pointer = pointer_path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise SystemExit(f"Cannot verify repository FP32 checkpoint LFS provenance: {exc}")
    marker = "oid sha256:"
    line = next((line for line in pointer.splitlines() if line.startswith(marker)), None)
    if line is None:
        raise SystemExit("Repository FP32 checkpoint is not a readable Git LFS pointer.")
    expected = line[len(marker):].strip()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise SystemExit("FP32 checkpoint LFS pointer contains an invalid SHA256.")
    try:
        manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read FP32 checkpoint manifest: {exc}")
    try:
        ptq_manifest = json.loads((ROOT / "experiments/ptq/manifest.json").read_text(encoding="utf-8"))
        export_manifest = json.loads((ROOT / "artifacts/openvino/export_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read supporting FP32 provenance records: {exc}")
    supporting_hashes = {
        ptq_manifest.get("source FP32 SHA256"),
        export_manifest.get("FP32 SHA256"),
    }
    provenance_matches = expected in supporting_hashes
    if (
        manifest.get("status") not in {"completed", "passed"}
        or manifest.get("class_mapping") != {"0": "ball", "1": "gawang", "2": "robot"}
        or not provenance_matches
    ):
        raise SystemExit("FP32 checkpoint manifest and repository LFS provenance do not agree.")
    return expected


def run_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "median", "min", "max", "std", "p50", "p90", "p95", "p99")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


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
        self.thread = threading.Thread(target=self._sample, name="pytorch-op3-resource-sampler", daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        start = time.perf_counter()
        while not self.stop_event.wait(self.interval):
            try:
                process_memory = self.process.memory_info()
                system_memory = psutil.virtual_memory()
                self.samples.append({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.perf_counter() - start,
                    "monotonic_ns": time.perf_counter_ns(),
                    "process_cpu_percent": self.process.cpu_percent(interval=None),
                    "system_cpu_percent": psutil.cpu_percent(interval=None),
                    "per_core_cpu_percent": json.dumps(psutil.cpu_percent(interval=None, percpu=True)),
                    "rss_bytes": process_memory.rss,
                    "vms_bytes": process_memory.vms,
                    "system_memory_used_bytes": system_memory.used,
                    "system_memory_available_bytes": system_memory.available,
                    "system_memory_percent": system_memory.percent,
                })
            except (psutil.Error, OSError):
                continue

    def stop(self) -> list[dict[str, Any]]:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=max(1.0, self.interval * 3))
        return self.samples


def resource_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "process_cpu_percent", "system_cpu_percent", "rss_bytes", "vms_bytes",
        "system_memory_used_bytes", "system_memory_available_bytes", "system_memory_percent",
    )
    result: dict[str, Any] = {"sample_count": len(samples)}
    for field in fields:
        result[field] = stats([float(sample[field]) for sample in samples if sample.get(field) is not None])
    if samples:
        per_core = [json.loads(sample["per_core_cpu_percent"]) for sample in samples]
        matrix = np.asarray(per_core, dtype=np.float64)
        result["per_core_cpu_percent"] = {
            "mean_by_core": np.mean(matrix, axis=0).tolist(),
            "peak_by_core": np.max(matrix, axis=0).tolist(),
            "p95_by_core": np.percentile(matrix, 95, axis=0).tolist(),
        }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_resource_plot(path: Path, samples: list[dict[str, Any]], label: str) -> None:
    """Draw a compact evidence chart directly from actual sampler rows."""
    if len(samples) < 2:
        return
    width, height = 1400, 760
    canvas = np.full((height, width, 3), (25, 28, 34), dtype=np.uint8)
    cv2.putText(canvas, f"OP3 resource utilization - {label}", (48, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
    panels = ((95, 115, 1280, 360, "Process CPU (%)", "process_cpu_percent", 1.0),
              (95, 435, 1280, 680, "Process RSS (MiB)", "rss_bytes", 1024.0 * 1024.0))
    times = np.asarray([float(item["elapsed_seconds"]) for item in samples])
    for top, left, right, bottom, title, key, divisor in panels:
        cv2.rectangle(canvas, (left, top), (right, bottom), (80, 85, 95), 1)
        cv2.putText(canvas, title, (left, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2)
        values = np.asarray([float(item[key]) / divisor for item in samples])
        low = min(0.0, float(values.min()))
        high = max(100.0 if key == "process_cpu_percent" else low + 1.0, float(values.max()) * 1.08)
        points = []
        for timestamp, value in zip(times, values):
            x = left + int((timestamp / max(times[-1], 1e-6)) * (right - left))
            y = bottom - int(((value - low) / max(high - low, 1e-6)) * (bottom - top))
            points.append((x, y))
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (65, 205, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"mean {values.mean():.1f} | p95 {np.percentile(values, 95):.1f} | peak {values.max():.1f}",
                    (left + 12, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Failed to write resource evidence image {path}")


def main() -> int:
    args = parse_args()
    if args.device.lower() != "cpu":
        raise SystemExit("Official ROBOTIS OP3 benchmark requires --device cpu.")
    if args.imgsz != 640 or args.warmup < 30:
        raise SystemExit("Use the fixed 640 input and at least 30 warmup inferences.")
    if args.mode == "synthetic" and args.iterations < 300:
        raise SystemExit("Use at least 300 timed synthetic iterations.")
    if args.mode == "camera" and not args.camera_device:
        raise SystemExit("Camera mode requires an audited --camera-device path.")
    if not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
        raise SystemExit("confidence and IoU thresholds must be in [0,1].")

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"FP32 checkpoint not found: {model_path}")
    expected_sha = get_expected_checkpoint_hash()
    model_sha = sha256_file(model_path)
    if model_sha != expected_sha:
        raise SystemExit("FP32 checkpoint SHA256 does not match the repository's frozen checkpoint.")

    run_timestamp = datetime.now(timezone.utc).isoformat()
    model_load_start = time.perf_counter_ns()
    model = YOLO(str(model_path), task="detect")
    model_load_ms = (time.perf_counter_ns() - model_load_start) / 1e6
    names = {int(key): str(value) for key, value in model.names.items()}
    if names != EXPECTED_NAMES:
        raise SystemExit(f"FP32 checkpoint class mapping mismatch: expected {EXPECTED_NAMES}, got {names}")
    model.model.eval()
    model.to("cpu")
    if torch.cuda.is_available():
        # CUDA may be present but is deliberately not selected by this run.
        selected_device = "cpu"
    else:
        selected_device = "cpu"
    synthetic = np.random.default_rng(42).integers(0, 256, (args.imgsz, args.imgsz, 3), dtype=np.uint8)
    capture: cv2.VideoCapture | None = None
    if args.mode == "camera":
        capture = cv2.VideoCapture(args.camera_device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise SystemExit(f"Unable to open audited camera {args.camera_device}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def predict(frame: np.ndarray) -> tuple[Any, dict[str, float], list[dict[str, Any]]]:
        result = model.predict(
            source=frame, imgsz=args.imgsz, batch=1, device=selected_device,
            conf=args.confidence, iou=args.iou, verbose=False, save=False, show=False,
        )[0]
        speed = {key: float(value) for key, value in result.speed.items()}
        if not all(np.isfinite(value) for value in speed.values()):
            raise RuntimeError("Ultralytics reported non-finite stage timing.")
        boxes = result.boxes
        detections = []
        for xyxy, confidence, class_tensor in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            class_id = int(class_tensor)
            detections.append({
                "class_id": class_id,
                "class_name": names[class_id],
                "confidence": float(confidence),
                "x1": float(xyxy[0]), "y1": float(xyxy[1]),
                "x2": float(xyxy[2]), "y2": float(xyxy[3]),
            })
        if not all(np.isfinite([item["confidence"], item["x1"], item["y1"], item["x2"], item["y2"]]).all() for item in detections):
            raise RuntimeError("Detection output contains NaN or Inf.")
        return result, speed, detections

    rows: list[dict[str, Any]] = []
    last_result = None
    frame_shape = None
    received = 0
    detection_total = 0
    sampler: ResourceSampler | None = None
    try:
        warm = 0
        while warm < args.warmup:
            if capture:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera read failed during warmup.")
                received += 1
            else:
                frame = synthetic
            predict(frame)
            warm += 1

        sampler = ResourceSampler(args.resource_interval)
        sampler.start()
        started_ns = time.perf_counter_ns()
        deadline = time.monotonic() + args.seconds if capture else None
        failures = 0
        processed = 0
        while True:
            if capture and deadline is not None and time.monotonic() >= deadline:
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
                    failures += 1
                    if failures >= 5:
                        break
                    continue
                failures = 0
                received += 1
            else:
                frame = synthetic
            last_result, speed, detections = predict(frame)
            frame_shape = list(frame.shape)
            loop_wall = (time.perf_counter_ns() - loop_start) / 1e6
            processed += 1
            detection_total += len(detections)
            rows.append({
                "iteration": processed,
                "capture_ms": capture_ms,
                "preprocess_ms": speed.get("preprocess", 0.0),
                "inference_ms": speed.get("inference", 0.0),
                "postprocess_ms": speed.get("postprocess", 0.0),
                "model_call_wall_ms": loop_wall - capture_ms,
                "end_to_end_wall_ms": loop_wall,
                "detection_count": len(detections),
                "detections": json.dumps(detections, separators=(",", ":")),
            })
        elapsed = (time.perf_counter_ns() - started_ns) / 1e9
        resource_samples = sampler.stop()
    finally:
        if capture:
            capture.release()
        if sampler is not None and sampler.thread is not None and sampler.thread.is_alive():
            sampler.stop()

    if not rows or last_result is None:
        raise SystemExit("No timed frames completed.")
    prefix = "benchmark_fp32_pytorch" if args.mode == "synthetic" else "camera_benchmark_fp32_pytorch"
    json_path = OUTPUT_DIR / f"{prefix}.json"
    csv_path = OUTPUT_DIR / f"{prefix}.csv"
    resources_path = OUTPUT_DIR / ("resources_fp32_pytorch.csv" if args.mode == "synthetic" else "resources_camera_benchmark_fp32_pytorch.csv")
    write_csv(csv_path, rows)
    write_csv(resources_path, resource_samples)
    timing = {field: stats([float(item[field]) for item in rows]) for field in (
        "capture_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "model_call_wall_ms", "end_to_end_wall_ms",
    )}
    resources = resource_summary(resource_samples)
    memory = psutil.virtual_memory()
    try:
        cpu_model = next(line.split(":", 1)[1].strip() for line in run_command(["lscpu"]).splitlines() if line.strip().startswith("Model name:"))
    except (AttributeError, StopIteration):
        cpu_model = platform.processor()
    report = {
        "timestamp_utc": run_timestamp,
        "git_commit": run_command(["git", "rev-parse", "HEAD"]),
        "git_branch": run_command(["git", "branch", "--show-current"]),
        "hostname": platform.node(),
        "cpu_model": cpu_model,
        "system_ram_total_bytes": memory.total,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "command": " ".join([sys.executable, *sys.argv]),
        "model": {
            "name": "YOLO11n FP32 native PyTorch",
            "path": str(model_path),
            "sha256": model_sha,
            "size_bytes": model_path.stat().st_size,
            "repository_checkpoint_manifest": str(MODEL_MANIFEST.relative_to(ROOT)),
            "provenance": "frozen repository FP32 best.pt; SHA256 matches LFS pointer and completed manifest",
            "quantization_method": "none (FP32); no export or calibration",
            "class_mapping": {str(key): value for key, value in names.items()},
        },
        "experimental_setup": {
            "mode": args.mode,
            "framework": "PyTorch + Ultralytics, native checkpoint inference",
            "openvino_used": False,
            "device": "CPU",
            "torch_cuda_available_but_unused": bool(torch.cuda.is_available()),
            "torch_num_threads": torch.get_num_threads(),
            "input_resolution": [args.imgsz, args.imgsz],
            "batch": 1,
            "warmups": args.warmup,
            "iterations_requested": args.iterations if args.mode == "synthetic" else None,
            "iterations_processed": len(rows),
            "duration_requested_seconds": args.seconds if args.mode == "camera" else None,
            "duration_measured_seconds": elapsed,
            "camera_device": args.camera_device,
            "camera_frames_received_including_warmup": received if capture is not None else None,
            "camera_frames_processed": len(rows) if capture is not None else None,
            "camera_dropped_frames": "unknown (V4L2/OpenCV capture has no source sequence number)" if capture is not None else None,
            "camera_frame_shape": frame_shape,
            "confidence_threshold": args.confidence,
            "iou_threshold": args.iou,
            "model_load_time_ms_excluded": model_load_ms,
            "resource_sample_interval_seconds": args.resource_interval,
            "visualization_enabled_during_measurement": False,
        },
        "latency_ms": timing,
        "fps_from_mean_inference_latency": 1000.0 / timing["inference_ms"]["mean"] if timing["inference_ms"]["mean"] else None,
        "effective_end_to_end_fps": len(rows) / elapsed if elapsed else None,
        "resource_summary": resources,
        "detection_observations": {
            "total_over_processed_frames": detection_total,
            "mean_per_frame": detection_total / len(rows),
            "last_frame_count": len(json.loads(rows[-1]["detections"])),
            "accuracy_metrics": "not computed; live/synthetic frames have no ground truth",
        },
        "raw_measurements_csv": str(csv_path.relative_to(ROOT)),
        "raw_resource_samples_csv": str(resources_path.relative_to(ROOT)),
        "model_load_time_excluded_from_latency": True,
        "official_minimums_met": args.warmup >= 30 and (args.mode != "synthetic" or args.iterations >= 300),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_resource_plot(args.evidence_dir / f"resource_utilization_{prefix}.png", resource_samples, "native PyTorch FP32")
    if args.mode == "camera":
        try:
            annotated = last_result.plot()
            evidence_path = args.evidence_dir / "live_detection_fp32_pytorch.png"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(evidence_path), annotated)
            report["detection_evidence_image"] = str(evidence_path.relative_to(ROOT))
            json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"Detection evidence image could not be rendered: {type(exc).__name__}: {exc}")
    print(f"Wrote {json_path.relative_to(ROOT)}, {csv_path.relative_to(ROOT)}, {resources_path.relative_to(ROOT)}")
    print(f"mean inference={timing['inference_ms']['mean']:.3f} ms, p95={timing['inference_ms']['p95']:.3f} ms")
    print(f"inference FPS={report['fps_from_mean_inference_latency']:.2f}, effective FPS={report['effective_end_to_end_fps']:.2f}")
    print(f"CPU mean={resources['process_cpu_percent']['mean']}, RSS peak={resources['rss_bytes']['max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
