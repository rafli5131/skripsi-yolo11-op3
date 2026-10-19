"""CPU OpenVINO YOLO11 image subscriber for an isolated thesis ROS workspace."""

from __future__ import annotations

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
import psutil
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from .measurement import RunRecorder, temperatures_c


CLASS_NAMES = {0: "ball", 1: "gawang", 2: "robot"}


def percentile_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "median", "min", "max", "std", "p50", "p90", "p95", "p99")}
    values_np = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values_np)),
        "median": float(np.median(values_np)),
        "min": float(np.min(values_np)),
        "max": float(np.max(values_np)),
        "std": float(np.std(values_np, ddof=1)) if len(values_np) > 1 else None,
        "p50": float(np.percentile(values_np, 50)),
        "p90": float(np.percentile(values_np, 90)),
        "p95": float(np.percentile(values_np, 95)),
        "p99": float(np.percentile(values_np, 99)),
    }


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized_width, resized_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - resized_width) / 2.0, (size - resized_height) / 2.0
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, scale, left, top


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while order.size and len(kept) < 100:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_current = max(0.0, float(boxes[current, 2] - boxes[current, 0])) * max(
            0.0, float(boxes[current, 3] - boxes[current, 1])
        )
        areas_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
            0.0, boxes[rest, 3] - boxes[rest, 1]
        )
        iou = intersection / np.maximum(area_current + areas_rest - intersection, 1e-9)
        order = rest[iou <= iou_threshold]
    return kept


def postprocess(
    output: np.ndarray,
    original_shape: tuple[int, int],
    transform: tuple[float, int, int],
    confidence_threshold: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    prediction = np.asarray(output)
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise RuntimeError(f"Expected rank-2 YOLO output, got {prediction.shape}")
    if prediction.shape[0] == len(CLASS_NAMES) + 4:
        rows = prediction.T
    elif prediction.shape[1] == len(CLASS_NAMES) + 4:
        rows = prediction
    else:
        raise RuntimeError(f"Expected 7 YOLO output values per proposal, got {prediction.shape}")

    class_scores = rows[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(rows)), class_ids]
    chosen = np.flatnonzero(scores >= confidence_threshold)
    if not chosen.size:
        return []

    boxes = rows[chosen, :4].astype(np.float32, copy=True)
    boxes[:, 0] -= boxes[:, 2] / 2.0
    boxes[:, 1] -= boxes[:, 3] / 2.0
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    scale, pad_left, pad_top = transform
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
    image_height, image_width = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height)

    detections: list[dict[str, Any]] = []
    for class_id, class_name in CLASS_NAMES.items():
        class_local = np.flatnonzero(class_ids[chosen] == class_id)
        if not class_local.size:
            continue
        for kept in nms_indices(boxes[class_local], scores[chosen[class_local]], iou_threshold):
            index = int(class_local[kept])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(scores[chosen[index]]),
                    "x1": float(boxes[index, 0]),
                    "y1": float(boxes[index, 1]),
                    "x2": float(boxes[index, 2]),
                    "y2": float(boxes[index, 3]),
                }
            )
    detections.sort(key=lambda item: -item["confidence"])
    return detections[:300]


class Yolo11Op3Vision(Node):
    def __init__(self) -> None:
        super().__init__("yolo11_op3_vision")
        self.declare_parameter("model_path", "")
        self.declare_parameter("input_topic", "/image_raw")
        self.declare_parameter("image_qos_reliability", "best_effort")
        self.declare_parameter("image_qos_depth", 5)
        self.declare_parameter("device", "CPU")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("debug_image_topic", "/vision/debug_image")
        self.declare_parameter("benchmark_seconds", 0.0)
        self.declare_parameter("warmup_iterations", 30)
        self.declare_parameter("resource_interval_seconds", 0.2)
        self.declare_parameter("output_dir", "")
        self.declare_parameter("result_stem", "ros2_benchmark")
        self.declare_parameter("evidence_image_path", "")
        self.declare_parameter("warmup_camera_frames", 30)
        self.declare_parameter("publish_benchmark", False)
        self.declare_parameter("benchmark_topic", "/vision/benchmark")
        self.declare_parameter("experiment_condition", "functional")
        self.declare_parameter("run_id", "run_01")
        self.declare_parameter("final_experiment", False)
        self.declare_parameter("thermal_limit_c", 90.0)

        self.model_path = Path(self.get_parameter("model_path").value).expanduser()
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.image_qos_reliability = str(self.get_parameter("image_qos_reliability").value)
        self.image_qos_depth = int(self.get_parameter("image_qos_depth").value)
        if self.image_qos_reliability not in {"best_effort", "reliable"} or self.image_qos_depth < 1:
            raise ValueError("Image QoS requires best_effort/reliable and depth >= 1")
        image_qos = QoSProfile(
            depth=self.image_qos_depth,
            reliability=(
                ReliabilityPolicy.RELIABLE
                if self.image_qos_reliability == "reliable"
                else ReliabilityPolicy.BEST_EFFORT
            ),
        )
        self.device = str(self.get_parameter("device").value).upper()
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.benchmark_seconds = float(self.get_parameter("benchmark_seconds").value)
        self.warmup_iterations = int(self.get_parameter("warmup_iterations").value)
        self.resource_interval = float(self.get_parameter("resource_interval_seconds").value)
        output_dir_value = str(self.get_parameter("output_dir").value).strip()
        self.output_dir = Path(output_dir_value).expanduser() if output_dir_value else None
        self.result_stem = str(self.get_parameter("result_stem").value)
        self.evidence_image_path = str(self.get_parameter("evidence_image_path").value)
        self.warmup_camera_frames = int(self.get_parameter("warmup_camera_frames").value)
        self.publish_benchmark = bool(self.get_parameter("publish_benchmark").value)
        self.benchmark_topic = str(self.get_parameter("benchmark_topic").value)
        self.condition = str(self.get_parameter("experiment_condition").value)
        self.run_id = str(self.get_parameter("run_id").value)
        self.final_experiment = bool(self.get_parameter("final_experiment").value)
        self.thermal_limit_c = float(self.get_parameter("thermal_limit_c").value)
        if self.condition not in {"diam", "berjalan", "functional", "stability"}:
            raise ValueError("experiment_condition must be diam, berjalan, functional, or stability")
        if self.warmup_camera_frames < 0 or self.resource_interval <= 0 or self.benchmark_seconds < 0:
            raise ValueError("Invalid camera warmup, resource interval, or duration")
        if self.final_experiment and (self.publish_debug_image or self.evidence_image_path):
            raise ValueError("Final experiments require debug/evidence rendering OFF")
        initial_temperatures = temperatures_c()
        if self.benchmark_seconds > 0 and initial_temperatures and max(initial_temperatures) >= 85:
            raise RuntimeError("Runtime test not started: initial temperature >= 85 C")

        if self.device != "CPU":
            raise RuntimeError("The OP3 study uses OpenVINO CPU inference only.")
        if self.imgsz != 640:
            raise RuntimeError("This study fixes the model input at 640x640.")
        if self.warmup_iterations < 30:
            raise RuntimeError("Use at least 30 warmup inferences.")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model XML does not exist: {self.model_path}")
        model_bin = self.model_path.with_suffix(".bin")
        if not model_bin.is_file():
            raise FileNotFoundError(f"Model BIN does not exist: {model_bin}")
        if self._is_lfs_pointer(self.model_path) or self._is_lfs_pointer(model_bin):
            raise RuntimeError(f"Model pair is still a Git LFS pointer: {self.model_path}")
        if self.benchmark_seconds > 0 and self.output_dir is None:
            raise RuntimeError("Set output_dir when benchmark_seconds is greater than zero.")

        try:
            import openvino as ov
        except ImportError as error:
            raise RuntimeError("OpenVINO Python is missing from this ROS runtime.") from error

        self.ov_version = ov.__version__
        core = ov.Core()
        if "CPU" not in core.available_devices:
            raise RuntimeError(f"OpenVINO CPU is unavailable: {core.available_devices}")
        model = core.read_model(str(self.model_path))
        model_input = model.input(0)
        input_shape = model_input.get_partial_shape()
        expected_shape = [1, 3, self.imgsz, self.imgsz]
        if not input_shape.is_static or list(input_shape.to_shape()) != expected_shape:
            raise RuntimeError(f"Expected model input {expected_shape}, got {input_shape}")
        compiled = core.compile_model(model, "CPU")
        self.request = compiled.create_infer_request()
        self.input_port = compiled.input(0)
        self.output_port = compiled.output(0)
        warmup = np.zeros(expected_shape, dtype=np.float32)
        for _ in range(self.warmup_iterations):
            self.request.infer({self.input_port: warmup})
        self.output_shape = list(self.request.get_output_tensor(0).data.shape)
        if not np.isfinite(self.request.get_output_tensor(0).data).all():
            raise RuntimeError("Model warmup produced NaN/Inf output.")

        self.bridge = CvBridge()
        self.detection_publisher = self.create_publisher(Detection2DArray, "/vision/detections", 10)
        self.debug_publisher = (
            self.create_publisher(Image, self.debug_image_topic, 1)
            if self.publish_debug_image
            else None
        )
        self.subscription = self.create_subscription(
            Image, self.input_topic, self.on_image, image_qos
        )
        self.frame_rows: list[dict[str, Any]] = []
        self.resource_rows: list[dict[str, Any]] = []
        self.process = psutil.Process(os.getpid())
        self.sample_stop = threading.Event()
        self.sample_thread: threading.Thread | None = None
        self.started_at = datetime.now(timezone.utc)
        self.run_started: float | None = None
        self.recorder: RunRecorder | None = None
        self.camera_warmups_done = 0
        self.completed = False
        self.thermal_abort = False
        self.rss_start = None
        self.measured_received = 0
        self.sample_no = 0
        self.telemetry_publisher = self.create_publisher(String, self.benchmark_topic, 100) if self.publish_benchmark else None
        self.frames_received = 0
        self.exceptions = 0
        self.evidence_saved = False
        self.finalized = False

        if self.benchmark_seconds > 0:
            self.stop_timer = self.create_timer(0.05, self._check_duration)
        else:
            self.stop_timer = None

        self.get_logger().info(
            f"OpenVINO {self.ov_version} on CPU; model={self.model_path}; "
            f"input={self.input_topic}; output=/vision/detections; "
            f"image_qos={self.image_qos_reliability}, depth={self.image_qos_depth}"
        )

    @staticmethod
    def _is_lfs_pointer(path: Path) -> bool:
        try:
            return path.open("rb").read(64).startswith(b"version https://git-lfs.github.com/spec/v1")
        except OSError:
            return False

    def _start_sampler(self) -> None:
        self.process.cpu_percent(interval=None)
        self.sample_thread = threading.Thread(target=self._sample_resources, daemon=True)
        self.sample_thread.start()

    def _sample_resources(self) -> None:
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        while not self.sample_stop.wait(self.resource_interval):
            try:
                memory = self.process.memory_info()
                system_memory = psutil.virtual_memory()
                self.sample_no += 1
                cpu = self.process.cpu_percent(interval=None)
                system_cpu = psutil.cpu_percent(interval=None)
                thermal = temperatures_c()
                self.recorder.resource(
                    {
                        "sample_no": self.sample_no,
                        "run_id": self.run_id,
                        "condition": self.condition,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": time.perf_counter() - self.run_started,
                        "cpu_usage_percent": system_cpu,
                        "memory_used_mb": system_memory.used / 1_000_000,
                        "memory_usage_percent": system_memory.percent,
                        "detector_process_cpu_percent": cpu,
                        "detector_rss_mb": memory.rss / 1_000_000,
                        "process_cpu_percent": cpu,
                        "system_cpu_percent": system_cpu,
                        "per_core_cpu_percent": json.dumps(psutil.cpu_percent(interval=None, percpu=True)),
                        "rss_bytes": memory.rss,
                        "vms_bytes": memory.vms,
                        "system_memory_used_bytes": system_memory.used,
                        "system_memory_available_bytes": system_memory.available,
                        "temperature_c": max(thermal) if thermal else "",
                    }
                )
                if thermal and max(thermal) >= self.thermal_limit_c:
                    self.thermal_abort = True
                    self.sample_stop.set()
            except (psutil.Error, OSError):
                continue

    def _begin_measurement(self):
        self.recorder = RunRecorder(self.output_dir, self.result_stem, self.run_id, self.condition, self.final_experiment)
        self.rss_start = self.process.memory_info().rss
        self.run_started = time.perf_counter()
        self._start_sampler()
        self.started_at = datetime.now(timezone.utc)
        self.get_logger().info(f"Measurement started: {self.condition}/{self.run_id}; camera warmup={self.camera_warmups_done}")

    def _annotate(self, frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
        annotated = frame.copy()
        colors = ((50, 220, 50), (0, 180, 255), (255, 100, 40))
        for item in detections:
            x1, y1, x2, y2 = (int(round(item[key])) for key in ("x1", "y1", "x2", "y2"))
            color = colors[int(item["class_id"]) % len(colors)]
            label = f"{item['class_name']} {item['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return annotated

    def on_image(self, message: Image) -> None:
        callback_start = time.perf_counter_ns()
        measuring = self.benchmark_seconds > 0 and self.camera_warmups_done >= self.warmup_camera_frames
        if measuring and self.run_started is None:
            self._begin_measurement()
            callback_start = time.perf_counter_ns()
        if measuring and self._check_duration():
            return
        self.frames_received += 1
        if measuring:
            self.measured_received += 1
        row = {"run_id": self.run_id, "condition": self.condition,
               "frame_index": self.measured_received if measuring else self.frames_received,
               "timestamp": datetime.now(timezone.utc).isoformat(), "error": "",
               "detection_count": 0, "preprocess_ms": None, "inference_ms": None,
               "postprocess_ms": None, "latency_ms": None, "image_conversion_ms": None,
               "callback_end_to_end_ms": None, "camera_header_stamp_ns": None,
               "camera_delivery_ms": None}
        try:
            phase = time.perf_counter_ns()
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            preprocess_start = time.perf_counter_ns()
            tensor, scale, pad_left, pad_top = letterbox(frame, self.imgsz)
            preprocess_ms = (time.perf_counter_ns() - callback_start) / 1e6
            row["preprocess_ms"] = preprocess_ms
            row["image_conversion_ms"] = (preprocess_start - phase) / 1e6

            inference_start = time.perf_counter_ns()
            self.request.infer({self.input_port: tensor})
            output = np.array(self.request.get_output_tensor(0).data, copy=True)
            inference_ms = (time.perf_counter_ns() - inference_start) / 1e6
            row["inference_ms"] = inference_ms
            if not np.isfinite(output).all():
                raise RuntimeError("Model output contains NaN/Inf.")

            postprocess_start = time.perf_counter_ns()
            detections = postprocess(
                output,
                frame.shape[:2],
                (scale, pad_left, pad_top),
                self.confidence_threshold,
                self.iou_threshold,
            )
            detection_ready = time.perf_counter_ns()
            postprocess_ms = (detection_ready - postprocess_start) / 1e6
            row["postprocess_ms"] = postprocess_ms
            row["latency_ms"] = (detection_ready - callback_start) / 1e6
            row["detection_count"] = len(detections)

            array_message = Detection2DArray()
            array_message.header = message.header
            for item in detections:
                detection = Detection2D()
                detection.header = message.header
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = f"{item['class_id']}:{item['class_name']}"
                hypothesis.hypothesis.score = item["confidence"]
                detection.results.append(hypothesis)
                detection.bbox.center.position.x = (item["x1"] + item["x2"]) / 2.0
                detection.bbox.center.position.y = (item["y1"] + item["y2"]) / 2.0
                detection.bbox.center.theta = 0.0
                detection.bbox.size_x = max(0.0, item["x2"] - item["x1"])
                detection.bbox.size_y = max(0.0, item["y2"] - item["y1"])
                array_message.detections.append(detection)
            self.detection_publisher.publish(array_message)

            annotated = None
            if self.debug_publisher is not None or (self.evidence_image_path and not self.evidence_saved):
                annotated = self._annotate(frame, detections)
            if self.debug_publisher is not None and annotated is not None:
                debug_message = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                debug_message.header = message.header
                self.debug_publisher.publish(debug_message)
            if self.evidence_image_path and not self.evidence_saved and annotated is not None:
                evidence_path = Path(self.evidence_image_path).expanduser()
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                self.evidence_saved = bool(cv2.imwrite(str(evidence_path), annotated))

            callback_ms = (time.perf_counter_ns() - callback_start) / 1e6
            header_stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
            now_ns = self.get_clock().now().nanoseconds
            delivery_ms = (now_ns - header_stamp_ns) / 1e6 if header_stamp_ns else None
            row.update(callback_end_to_end_ms=callback_ms, camera_header_stamp_ns=header_stamp_ns or None, camera_delivery_ms=delivery_ms)
            if not measuring:
                self.camera_warmups_done += 1
        except Exception as error:  # Keep the camera callback alive and record failures.
            self.exceptions += 1
            row["error"] = str(error)
            self.get_logger().error(f"Image callback failed: {error}")
        finally:
            if measuring:
                self.recorder.frame(row)
            if self.telemetry_publisher:
                telemetry = {**row, "warmup": not measuring, "callback_latency_ms": row["callback_end_to_end_ms"],
                             "image_stamp_ns": row["camera_header_stamp_ns"]}
                self.telemetry_publisher.publish(String(data=json.dumps(telemetry)))

    def _check_duration(self) -> bool:
        if self.run_started is not None and (self.thermal_abort or time.perf_counter() - self.run_started >= self.benchmark_seconds):
            self.completed = not self.thermal_abort
            self._finalize()
            rclpy.shutdown()
            return True
        return False

    def _repo_commit(self) -> str | None:
        for parent in self.model_path.resolve().parents:
            if (parent / ".git").exists():
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=parent, capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        return None

    def _summary(self, field: str) -> dict[str, float | None]:
        return percentile_stats([float(row[field]) for row in self.frame_rows if not row.get("error") and row.get(field) not in (None, "")])

    def _finalize(self) -> None:
        if self.finalized:
            return
        self.finalized = True
        elapsed = max(0.0, time.perf_counter() - self.run_started) if self.run_started is not None else 0.0
        rss_end = self.process.memory_info().rss
        self.sample_stop.set()
        if self.sample_thread is not None:
            self.sample_thread.join(timeout=max(1.0, self.resource_interval * 2))
        if self.recorder is None:
            return

        self.recorder.close()
        timing_path = self.recorder.frame_path
        resource_path = self.recorder.resource_path
        self.frame_rows, self.resource_rows = self.recorder.rows()
        successes = [row for row in self.frame_rows if not row.get("error")]

        process_cpu = [float(row["process_cpu_percent"]) for row in self.resource_rows]
        system_cpu = [float(row["system_cpu_percent"]) for row in self.resource_rows]
        rss_mib = [float(row["rss_bytes"]) / (1024**2) for row in self.resource_rows]
        memory = psutil.virtual_memory()
        report = {
            "run_id": self.run_id,
            "condition": self.condition,
            "completed": self.completed,
            "thermal_abort": self.thermal_abort,
            "statistics_ddof": 1,
            "rss_start_bytes": self.rss_start,
            "rss_end_bytes": rss_end,
            "rss_growth_bytes": rss_end - self.rss_start if self.rss_start is not None else None,
            "thermal_end_c": temperatures_c(),
            "timestamp_utc": self.started_at.isoformat(),
            "git_commit": self._repo_commit(),
            "hostname": platform.node(),
            "cpu_model": platform.processor(),
            "system_ram_total_bytes": memory.total,
            "openvino_version": self.ov_version,
            "python_version": platform.python_version(),
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "command": " ".join([sys.executable, *sys.argv]),
            "model": {
                "path": str(self.model_path.resolve()),
                "artifact_identity_verification": "not performed; user-provided model path",
                "size_bytes": self.model_path.stat().st_size,
                "input_shape": [1, 3, self.imgsz, self.imgsz],
                "output_shape": self.output_shape,
            },
            "setup": {
                "device": "CPU",
                "imgsz": self.imgsz,
                "batch": 1,
                "input_topic": self.input_topic,
                "image_qos_reliability": self.image_qos_reliability,
                "image_qos_depth": self.image_qos_depth,
                "warmup_inferences": self.warmup_iterations,
                "warmup_camera_frames": self.camera_warmups_done,
                "publish_benchmark": self.publish_benchmark,
                "benchmark_seconds_requested": self.benchmark_seconds,
                "benchmark_seconds_measured": elapsed,
                "resource_interval_seconds": self.resource_interval,
                "publish_debug_image": self.publish_debug_image,
                "evidence_image_path": self.evidence_image_path or None,
            },
            "results": {
                "frames_received": self.measured_received,
                "total_camera_frames_received_including_warmup": self.frames_received,
                "frames_processed": len(successes),
                "effective_fps": len(successes) / elapsed if elapsed > 0 else 0.0,
                "detection_count": sum(int(row["detection_count"]) for row in self.frame_rows),
                "exceptions": len(self.frame_rows) - len(successes),
                "total_exceptions_including_warmup": self.exceptions,
                "latency_ms": {
                    field: self._summary(field)
                    for field in (
                        "camera_delivery_ms",
                        "image_conversion_ms",
                        "preprocess_ms",
                        "inference_ms",
                        "postprocess_ms",
                        "callback_end_to_end_ms",
                        "latency_ms",
                    )
                },
                "cpu_percent": {
                    "process": percentile_stats(process_cpu),
                    "system": percentile_stats(system_cpu),
                },
                "rss_mib": percentile_stats(rss_mib),
                "resource_sample_count": len(self.resource_rows),
            },
            "raw_measurements": {
                "timing_csv": str(timing_path) if timing_path.exists() else None,
                "resource_csv": str(resource_path) if resource_path.exists() else None,
            },
        }
        json_path = self.recorder.report_path
        with json_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        self.get_logger().info(f"ROS benchmark results saved to {json_path}")

    def destroy_node(self):
        self._finalize()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = Yolo11Op3Vision()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"yolo11_op3_vision failed: {error}", file=sys.stderr)
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
