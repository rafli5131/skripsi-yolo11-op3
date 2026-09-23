"""Measure ROS 2 camera-to-detection performance and node resources."""

from __future__ import annotations

import argparse
import csv
from importlib.metadata import PackageNotFoundError, version as distribution_version
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

import numpy as np
import psutil
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from op3_model_audit import qat_validation_gate, ptq_validation_gate, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "experiments/hardware_op3"


def temperatures_c() -> list[float]:
    values: list[float] = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(int(path.read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return values


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "median", "min", "max", "std", "p50", "p90", "p95", "p99")}
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "std": float(np.std(data)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def package_version(name: str) -> str | None:
    try:
        return distribution_version(name)
    except PackageNotFoundError:
        return None


def detector_pid(explicit_pid: int | None) -> int:
    if explicit_pid:
        psutil.Process(explicit_pid)
        return explicit_pid
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "yolo11_openvino_node" in cmdline and "benchmark_op3_ros2.py" not in cmdline:
                return int(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    raise RuntimeError("Could not find yolo11_openvino_node; pass --node-pid explicitly.")


class NodeResourceSampler:
    def __init__(self, pid: int, interval: float) -> None:
        self.process = psutil.Process(pid)
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        psutil.cpu_percent(interval=None, percpu=True)
        self.process.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._sample, name="ros2-node-resource-sampler", daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        while not self.stopped.wait(self.interval):
            try:
                memory = self.process.memory_info()
                system_memory = psutil.virtual_memory()
                self.samples.append(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "monotonic_ns": time.perf_counter_ns(),
                        "node_pid": self.process.pid,
                        "process_cpu_percent": self.process.cpu_percent(interval=None),
                        "system_cpu_percent": psutil.cpu_percent(interval=None),
                        "per_core_cpu_percent": psutil.cpu_percent(interval=None, percpu=True),
                        "rss_bytes": memory.rss,
                        "vms_bytes": memory.vms,
                        "system_memory_used_bytes": system_memory.used,
                        "system_memory_available_bytes": system_memory.available,
                        "system_memory_percent": system_memory.percent,
                        "thermal_zone_temperatures_c": temperatures_c(),
                    }
                )
            except (psutil.Error, OSError):
                continue

    def stop(self) -> list[dict[str, Any]]:
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=max(1.0, self.interval * 3))
        return self.samples


def summarize_resources(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"sample_count": len(samples)}
    fields = (
        "process_cpu_percent", "system_cpu_percent", "rss_bytes", "vms_bytes",
        "system_memory_used_bytes", "system_memory_available_bytes", "system_memory_percent",
    )
    for key in fields:
        summary[key] = stats([float(sample[key]) for sample in samples if sample.get(key) is not None])
    if samples:
        first_rss, last_rss = samples[0]["rss_bytes"], samples[-1]["rss_bytes"]
        summary["rss_first_sample_bytes"] = first_rss
        summary["rss_last_sample_bytes"] = last_rss
        summary["rss_growth_bytes"] = last_rss - first_rss
        summary["rss_growth_percent"] = ((last_rss - first_rss) / first_rss * 100.0) if first_rss else None
        per_core = np.asarray([sample["per_core_cpu_percent"] for sample in samples], dtype=np.float64)
        summary["per_core_cpu_percent"] = {
            "mean_by_core": np.mean(per_core, axis=0).tolist(),
            "peak_by_core": np.max(per_core, axis=0).tolist(),
            "p95_by_core": np.percentile(per_core, 95, axis=0).tolist(),
        }
        thermal = [value for sample in samples for value in sample.get("thermal_zone_temperatures_c", [])]
        summary["thermal_zone_temperature_c"] = {
            "maximum_sampled": max(thermal) if thermal else None,
            "last_sample": samples[-1].get("thermal_zone_temperatures_c", []),
        }
    return summary


class BenchmarkSubscriber(Node):
    def __init__(self, warmups: int) -> None:
        super().__init__("op3_ros2_benchmark")
        self.warmups = warmups
        self.warmup_count = 0
        self.started_at: float | None = None
        self.rows: list[dict[str, Any]] = []
        self.create_subscription(String, "/vision/benchmark", self.callback, 100)

    def callback(self, message: String) -> None:
        try:
            sample = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if self.started_at is None:
            self.warmup_count += 1
            if self.warmup_count >= self.warmups:
                self.started_at = time.monotonic()
            return
        self.rows.append(sample)


def save_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            flattened = dict(row)
            for key, value in list(flattened.items()):
                if isinstance(value, (dict, list)):
                    flattened[key] = json.dumps(value, sort_keys=True)
            writer.writerow(flattened)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("fp32", "ptq_int8", "qat_int8"), default="fp32")
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--resource-interval", type=float, default=0.2)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--node-pid", type=int)
    parser.add_argument("--result-stem", default="ros2_benchmark")
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model_kind == "qat_int8" and not qat_validation_gate(ROOT)["accepted"]:
        raise SystemExit("BLOCKED: no validated QAT OpenVINO artifact")
    if args.model_kind == "ptq_int8" and not ptq_validation_gate(ROOT, args.model_xml)["accepted"]:
        raise SystemExit("BLOCKED: existing PTQ artifact failed provenance/hash validation")
    if args.warmup < 30:
        raise SystemExit("ROS 2 benchmark requires at least 30 warmup callbacks.")
    if args.result_stem == "stability_test" and args.seconds < 600:
        raise SystemExit("The camera stability test must run for at least 600 seconds.")
    initial_temperatures = temperatures_c()
    if args.result_stem == "stability_test" and initial_temperatures and max(initial_temperatures) >= 85.0:
        raise SystemExit(f"Stability test not started: unsafe initial thermal zone {max(initial_temperatures):.1f} C.")

    model_xml = args.model_xml.expanduser().resolve()
    model_bin = model_xml.with_suffix(".bin")
    if not model_xml.is_file() or not model_bin.is_file():
        raise SystemExit(f"OpenVINO IR XML/BIN pair not found: {model_xml} and {model_bin}")

    rclpy.init(args=None)
    subscriber = BenchmarkSubscriber(args.warmup)
    sampler: NodeResourceSampler | None = None
    samples: list[dict[str, Any]] = []
    thermal_abort = False
    detector_process_id: int | None = None
    launch_time = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(subscriber, timeout_sec=0.1)
            if subscriber.started_at is None:
                if time.monotonic() - launch_time > args.startup_timeout:
                    raise RuntimeError("No YOLO benchmark messages arrived after startup timeout.")
                continue
            if sampler is None:
                detector_process_id = detector_pid(args.node_pid)
                sampler = NodeResourceSampler(detector_process_id, args.resource_interval)
                sampler.start()
            if args.result_stem == "stability_test":
                temperatures = temperatures_c()
                if temperatures and max(temperatures) >= 90.0:
                    thermal_abort = True
                    break
            if time.monotonic() - subscriber.started_at >= args.seconds:
                break
    finally:
        if sampler:
            samples = sampler.stop()
        subscriber.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if subscriber.started_at is None or not subscriber.rows:
        raise RuntimeError("ROS 2 run did not produce measured callback samples.")
    measured_duration = time.monotonic() - subscriber.started_at
    successes = [row for row in subscriber.rows if not row.get("error")]
    failures = [row for row in subscriber.rows if row.get("error")]
    timing_fields = (
        "image_age_at_callback_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "callback_latency_ms"
    )
    latency = {
        field: stats([float(row[field]) for row in successes if row.get(field) is not None])
        for field in timing_fields
    }
    stamps = [int(row["image_stamp_ns"]) for row in subscriber.rows if int(row.get("image_stamp_ns", 0)) > 0]
    source_interval_s = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else None
    expected_source_frames = int(round(measured_duration * args.camera_fps))
    estimated_gaps = max(0, expected_source_frames - len(successes)) if source_interval_s else None
    resources = summarize_resources(samples)
    thermal_end = temperatures_c()
    cpu_model = None
    try:
        lscpu = subprocess.run(["lscpu"], text=True, capture_output=True, check=False, timeout=5).stdout
        cpu_model = next((line.split(":", 1)[1].strip() for line in lscpu.splitlines() if line.startswith("Model name:")), None)
    except (OSError, subprocess.TimeoutExpired):
        pass

    stem = args.result_stem
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{stem}.json"
    csv_path = OUTPUT_DIR / f"{stem}.csv"
    resource_path = OUTPUT_DIR / f"resources_{stem}.csv"
    resource_columns = [
        "timestamp_utc", "monotonic_ns", "node_pid", "process_cpu_percent", "system_cpu_percent",
        "per_core_cpu_percent", "rss_bytes", "vms_bytes", "system_memory_used_bytes",
        "system_memory_available_bytes", "system_memory_percent", "thermal_zone_temperatures_c",
    ]
    save_csv(resource_path, samples, resource_columns)
    event_columns = [
        "frame_index", "processed_frames", "exceptions", "image_stamp_ns", "image_age_at_callback_ms",
        "preprocess_ms", "inference_ms", "postprocess_ms", "callback_latency_ms", "detection_count", "error",
    ]
    save_csv(csv_path, subscriber.rows, event_columns)
    inference_mean = latency["inference_ms"]["mean"]
    memory = psutil.virtual_memory()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "hostname": platform.node(),
        "cpu_model": cpu_model,
        "system_ram_total_bytes": memory.total,
        "python_version": platform.python_version(),
        "openvino_version": package_version("openvino"),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "command": " ".join([sys.executable, *sys.argv]),
        "model": {
            "kind": args.model_kind,
            "xml_path": str(model_xml),
            "xml_sha256": sha256_file(model_xml),
            "bin_path": str(model_bin),
            "bin_sha256": sha256_file(model_bin),
        },
        "setup": {
            "device": "CPU",
            "input_resolution": [640, 640],
            "batch": 1,
            "image_topic": "/image_raw",
            "detection_topic": "/vision/detections",
            "debug_topic": "/vision/debug_image",
            "debug_image_enabled": False,
            "benchmark_topic": "/vision/benchmark",
            "warmup_callbacks": args.warmup,
            "warmup_callbacks_observed": subscriber.warmup_count,
            "duration_requested_seconds": args.seconds,
            "duration_measured_seconds": measured_duration,
            "resource_sample_interval_seconds": args.resource_interval,
            "camera_fps_configured": args.camera_fps,
            "camera_frames_estimated_dropped": estimated_gaps,
            "camera_drop_estimate_method": "configured source FPS * measured interval minus successful detector callbacks; estimate only",
            "frames_received_by_detector_callback": len(subscriber.rows) + subscriber.warmup_count,
            "frames_processed_successfully": len(successes),
            "exceptions": len(failures),
            "exception_messages": [row.get("error") for row in failures[:20]],
            "thermal_initial_c": initial_temperatures,
            "thermal_final_c": thermal_end,
            "thermal_abort": thermal_abort,
            "thermal_abort_threshold_c": 90.0 if stem == "stability_test" else None,
            "node_pid": detector_process_id,
        },
        "latency_ms": latency,
        "fps_from_mean_inference_latency": 1000.0 / inference_mean if inference_mean else None,
        "effective_end_to_end_fps": len(successes) / measured_duration if measured_duration else None,
        "resource_summary": resources,
        "accuracy_metrics": "not computed; live camera frames have no ground truth",
        "raw_measurements_csv": str(csv_path.relative_to(ROOT)),
        "raw_resource_samples_csv": str(resource_path.relative_to(ROOT)),
        "stability": {
            "completed_full_duration": stem == "stability_test" and not thermal_abort and measured_duration >= 600,
            "rss_start_bytes": resources.get("rss_first_sample_bytes"),
            "rss_end_bytes": resources.get("rss_last_sample_bytes"),
            "rss_growth_bytes": resources.get("rss_growth_bytes"),
            "rss_growth_percent": resources.get("rss_growth_percent"),
            "exceptions": len(failures),
            "estimated_dropped_frames": estimated_gaps,
        } if stem == "stability_test" else None,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)}, {csv_path.relative_to(ROOT)}, {resource_path.relative_to(ROOT)}")
    print(
        f"ROS frames={len(successes)} exceptions={len(failures)} inference mean={inference_mean:.3f} ms; "
        f"effective FPS={report['effective_end_to_end_fps']:.2f}"
    )
    print(
        f"node CPU mean={resources['process_cpu_percent']['mean']}%; "
        f"node peak RSS={resources['rss_bytes']['max']} bytes; thermal_abort={thermal_abort}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
