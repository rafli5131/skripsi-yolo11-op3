"""Capture a live C920 detection screenshot and render the measured ROS resource trace."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import psutil
from matplotlib.ticker import FormatStrFormatter

from benchmark_op3 import ResourceSampler, letterbox, percentile_stats, postprocess
from op3_model_audit import sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments/hardware_op3/evidence"
DEFAULT_MODEL = ROOT / ".cache/hardware_models/fp32/model.xml"
RESOURCE_SOURCE = ROOT / "experiments/hardware_op3/resources_stability_test.csv"
STABILITY_SOURCE = ROOT / "experiments/hardware_op3/stability_test.json"


def git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def draw_detection_card(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    captured_at: str,
    host: str,
    camera_device: str,
    model_sha: str,
    inference_ms: float,
    capture_duration_s: float,
    frame_count: int,
    resource_summary: dict[str, Any],
) -> np.ndarray:
    width, height = 1440, 900
    card = np.full((height, width, 3), (24, 29, 38), dtype=np.uint8)
    cv2.putText(card, "ROBOTIS OP3 | LIVE OBJECT DETECTION", (42, 57), cv2.FONT_HERSHEY_SIMPLEX, 1.05,
                (245, 248, 252), 2, cv2.LINE_AA)
    cv2.putText(card, "YOLO11n FP32 OpenVINO  /  CPU  /  640 x 640", (44, 88), cv2.FONT_HERSHEY_SIMPLEX,
                0.56, (165, 182, 201), 1, cv2.LINE_AA)

    image_x, image_y, image_w, image_h = 40, 112, 1020, 765
    frame_h, frame_w = frame.shape[:2]
    scale = min(image_w / frame_w, image_h / frame_h)
    shown_w, shown_h = int(round(frame_w * scale)), int(round(frame_h * scale))
    shown = cv2.resize(frame, (shown_w, shown_h), interpolation=cv2.INTER_AREA)
    colors = {0: (50, 225, 95), 1: (0, 190, 255), 2: (255, 137, 65)}
    for item in detections:
        color = colors.get(int(item["class_id"]), (240, 240, 240))
        x1, y1, x2, y2 = [int(round(float(item[key]) * scale)) for key in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(shown, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        label = f"{item['class_name']} {float(item['confidence']):.3f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        label_y = max(text_h + baseline + 4, y1)
        cv2.rectangle(shown, (x1, label_y - text_h - baseline - 7),
                      (x1 + text_w + 12, label_y + 2), color, -1)
        cv2.putText(shown, label, (x1 + 6, label_y - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (20, 25, 31), 2, cv2.LINE_AA)
    paste_x = image_x + (image_w - shown_w) // 2
    paste_y = image_y + (image_h - shown_h) // 2
    card[paste_y:paste_y + shown_h, paste_x:paste_x + shown_w] = shown
    cv2.rectangle(card, (image_x - 1, image_y - 1), (image_x + image_w, image_y + image_h),
                  (76, 88, 105), 1)

    panel_x, panel_y, panel_w, panel_h = 1090, 112, 310, 765
    cv2.rectangle(card, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  (35, 42, 54), -1)
    cv2.putText(card, "DETECTIONS", (panel_x + 22, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (245, 248, 252), 2, cv2.LINE_AA)
    cv2.putText(card, f"{len(detections)} object(s)", (panel_x + 22, panel_y + 77),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 225, 150), 2, cv2.LINE_AA)
    cursor_y = panel_y + 122
    if detections:
        for index, item in enumerate(detections[:8], start=1):
            color = colors.get(int(item["class_id"]), (240, 240, 240))
            cv2.circle(card, (panel_x + 29, cursor_y - 5), 6, color, -1, cv2.LINE_AA)
            cv2.putText(card, f"{index}. {item['class_name']}", (panel_x + 47, cursor_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 239, 245), 1, cv2.LINE_AA)
            cv2.putText(card, f"class {item['class_id']}  |  conf {item['confidence']:.3f}",
                        (panel_x + 47, cursor_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (171, 188, 207), 1, cv2.LINE_AA)
            cursor_y += 64
    else:
        cv2.putText(card, "No detections in selected frame", (panel_x + 22, cursor_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 198, 208), 1, cv2.LINE_AA)

    info_y = max(cursor_y + 22, panel_y + 360)
    cv2.line(card, (panel_x + 20, info_y - 25), (panel_x + panel_w - 20, info_y - 25), (74, 87, 104), 1)
    facts = [
        f"Host: {host}",
        f"Camera: {camera_device}",
        f"Observed frames: {frame_count}",
        f"Capture window: {capture_duration_s:.1f} s",
        f"Selected inference: {inference_ms:.2f} ms",
        f"Model SHA256: {model_sha[:16]}...",
        f"Captured: {captured_at}",
    ]
    for fact in facts:
        cv2.putText(card, fact, (panel_x + 22, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                    (186, 199, 214), 1, cv2.LINE_AA)
        info_y += 31
    cv2.line(card, (panel_x + 20, info_y - 8), (panel_x + panel_w - 20, info_y - 8), (74, 87, 104), 1)
    info_y += 20
    cv2.putText(card, "CAPTURE RESOURCES", (panel_x + 22, info_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (245, 248, 252), 1, cv2.LINE_AA)
    info_y += 29
    live_resource_lines = [
        f"Process CPU mean/P95: {resource_summary['process_cpu_percent']['mean']:.1f}/"
        f"{resource_summary['process_cpu_percent']['p95']:.1f}%",
        f"Process RSS mean/peak: {resource_summary['rss_bytes']['mean'] / 1048576:.1f}/"
        f"{resource_summary['rss_bytes']['max'] / 1048576:.1f} MiB",
        f"Samples: {resource_summary['resource_sample_count']}  (0.2 s interval)",
    ]
    for line in live_resource_lines:
        cv2.putText(card, line, (panel_x + 22, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (130, 220, 184), 1, cv2.LINE_AA)
        info_y += 25
    cv2.putText(card, "Frame and boxes are from live camera inference.", (42, 895),
                cv2.FONT_HERSHEY_SIMPLEX, 0.39, (150, 166, 185), 1, cv2.LINE_AA)
    return card


def render_resource_plot(stability: dict[str, Any]) -> Path:
    if not stability.get("stability", {}).get("completed_full_duration"):
        raise RuntimeError("The source ROS stability record does not certify a complete 10-minute run.")
    if not RESOURCE_SOURCE.is_file():
        raise FileNotFoundError(f"Missing measured resource samples: {RESOURCE_SOURCE}")
    monotonic_ns: list[int] = []
    process_cpu: list[float] = []
    system_cpu: list[float] = []
    rss_mib: list[float] = []
    system_memory_percent: list[float] = []
    with RESOURCE_SOURCE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            monotonic_ns.append(int(row["monotonic_ns"]))
            process_cpu.append(float(row["process_cpu_percent"]))
            system_cpu.append(float(row["system_cpu_percent"]))
            rss_mib.append(int(row["rss_bytes"]) / (1024 * 1024))
            system_memory_percent.append(float(row["system_memory_percent"]))
    # Anchor time to the first sample; monotonic timestamps are host-local.
    elapsed = [(value - monotonic_ns[0]) / 1e9 for value in monotonic_ns] if monotonic_ns else []
    if not elapsed:
        raise RuntimeError("The stability resource CSV contains no samples.")

    elapsed_s = np.asarray(elapsed, dtype=np.float64)
    process_cpu_values = np.asarray(process_cpu, dtype=np.float64)
    system_cpu_values = np.asarray(system_cpu, dtype=np.float64)
    rss_values = np.asarray(rss_mib, dtype=np.float64)
    system_memory_values = np.asarray(system_memory_percent, dtype=np.float64)
    # Plot 5-second means for readability; retain all 0.2-second samples in CSV.
    bucket_ids = np.floor(elapsed_s / 5.0).astype(np.int64)
    unique_buckets = np.unique(bucket_ids)
    bucket_centers = (unique_buckets + 0.5) * 5.0 / 60.0

    def bucket_means(values: np.ndarray) -> np.ndarray:
        return np.asarray([values[bucket_ids == bucket].mean() for bucket in unique_buckets])

    plot_process_cpu = bucket_means(process_cpu_values)
    plot_system_cpu = bucket_means(system_cpu_values)
    plot_rss = bucket_means(rss_values)
    plot_system_memory = bucket_means(system_memory_values)

    output = OUT_DIR / "resource_utilization_10min.png"
    duration_min = stability["setup"]["duration_measured_seconds"] / 60.0
    frame_count = stability["setup"]["frames_processed_successfully"]
    exceptions = stability["setup"]["exceptions"]
    fig, (ax_cpu, ax_mem) = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                                        gridspec_kw={"height_ratios": [1.2, 1]})
    fig.patch.set_facecolor("#f4f7fb")
    for ax in (ax_cpu, ax_mem):
        ax.set_facecolor("white")
        ax.grid(True, color="#dfe6ef", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color("#8795a8")
        ax.spines["bottom"].set_color("#8795a8")
    ax_cpu.plot(bucket_centers, plot_process_cpu, color="#2267b1", linewidth=1.5,
                label="Detector process CPU (5 s mean; aggregate cores)")
    ax_cpu.plot(bucket_centers, plot_system_cpu, color="#ee8b2d", linewidth=1.2, alpha=0.95,
                label="Whole-system CPU (%)")
    ax_cpu.axhline(float(stability["resource_summary"]["process_cpu_percent"]["mean"]),
                   color="#2267b1", linestyle="--", linewidth=1.2, alpha=0.8,
                   label=f"Detector mean {stability['resource_summary']['process_cpu_percent']['mean']:.1f}%")
    ax_cpu.set_ylabel("CPU utilization (%)")
    ax_cpu.legend(loc="upper right", frameon=True, framealpha=0.92, facecolor="white",
                  edgecolor="#dfe6ef", fontsize=9)
    ax_cpu.set_title("CPU load during camera-backed ROS 2 inference", loc="left", fontsize=13, weight="bold")

    ax_mem.plot(bucket_centers, plot_rss, color="#18875b", linewidth=1.5,
                label="Detector RSS (5 s mean, MiB)")
    ax_mem.set_ylabel("Detector RSS (MiB)")
    ax_mem.set_xlabel("Elapsed time (minutes)")
    rss_pad = max(0.5, float(np.ptp(rss_values)) * 0.25)
    ax_mem.set_ylim(float(rss_values.min()) - rss_pad, float(rss_values.max()) + rss_pad)
    ax_mem.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax_system_mem = ax_mem.twinx()
    ax_system_mem.plot(bucket_centers, plot_system_memory, color="#a4529a", linewidth=1.2, alpha=0.9,
                       label="System memory used (%)")
    ax_system_mem.set_ylabel("System memory used (%)")
    lines1, labels1 = ax_mem.get_legend_handles_labels()
    lines2, labels2 = ax_system_mem.get_legend_handles_labels()
    ax_mem.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True,
                  framealpha=0.92, facecolor="white", edgecolor="#dfe6ef", fontsize=9)
    ax_mem.set_title("Memory use: 0.2 s samples summarized as 5 s means", loc="left",
                     fontsize=13, weight="bold")
    fig.suptitle(
        f"YOLO11n FP32 OpenVINO on CPU | ROS 2 + C920 | {duration_min:.2f} min | "
        f"{frame_count:,} processed frames | {exceptions} exceptions",
        x=0.055, y=0.985, ha="left", fontsize=16, weight="bold", color="#1c2d42",
    )
    fig.text(0.055, 0.945,
             "Measured run: stability_test.json  •  Raw samples: resources_stability_test.csv  •  "
             f"Host: {stability['hostname']}  •  Peak sampled temperature: "
             f"{stability['resource_summary']['thermal_zone_temperature_c']['maximum_sampled']:.0f} °C",
             fontsize=9, color="#4e5e72")
    ax_mem.set_xlim(0, max(duration_min, bucket_centers[-1]))
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.92))
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera-device", default="/dev/video0")
    parser.add_argument("--capture-seconds", type=float, default=15.0)
    parser.add_argument("--resource-interval", type=float, default=0.2)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = json.loads((ROOT / "experiments/hardware_op3/model_inventory.json").read_text())
    if inventory["fp32_baseline"]["validity"] != "valid":
        raise SystemExit("Refusing evidence capture: FP32 model provenance is not marked valid.")
    xml_path = args.model_xml.expanduser().resolve()
    bin_path = xml_path.with_suffix(".bin")
    if not xml_path.is_file() or not bin_path.is_file():
        raise SystemExit(f"OpenVINO XML/BIN pair not found: {xml_path}, {bin_path}")
    expected_hash = inventory["fp32_baseline"]["openvino_xml"]["sha256"]
    model_hash = sha256_file(xml_path)
    if model_hash != expected_hash:
        raise SystemExit("Refusing evidence capture: model XML SHA256 does not match the audited FP32 artifact.")
    core = ov.Core()
    if "CPU" not in core.available_devices:
        raise SystemExit(f"OpenVINO CPU unavailable: {core.available_devices}")
    model = core.read_model(str(xml_path))
    compiled = core.compile_model(model, "CPU")
    infer = compiled.create_infer_request()
    input_port = compiled.input(0)
    if list(compiled.input(0).get_shape()) != [1, 3, 640, 640]:
        raise SystemExit(f"Unexpected model input shape: {compiled.input(0).get_shape()}")
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    for _ in range(30):
        infer.infer({input_port: dummy})

    capture = cv2.VideoCapture(args.camera_device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise SystemExit(f"Could not open audited camera device {args.camera_device}.")
    sampler = ResourceSampler(args.resource_interval)
    sampler.start()
    measurements: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    started = time.perf_counter()
    capture_started = datetime.now(timezone.utc).isoformat()
    try:
        while time.perf_counter() - started < args.capture_seconds:
            read_started_ns = time.perf_counter_ns()
            ok, frame = capture.read()
            capture_ms = (time.perf_counter_ns() - read_started_ns) / 1e6
            if not ok or frame is None:
                continue
            tensor, scale, pad_x, pad_y = letterbox(frame, 640)
            infer_started_ns = time.perf_counter_ns()
            infer.infer({input_port: tensor})
            inference_ms = (time.perf_counter_ns() - infer_started_ns) / 1e6
            raw = np.array(infer.get_output_tensor(0).data, copy=True)
            if not np.isfinite(raw).all():
                raise RuntimeError("OpenVINO output contains NaN or Inf during evidence capture.")
            detections = postprocess(raw, frame.shape[:2], (scale, pad_x, pad_y), args.confidence, args.iou)
            row = {
                "frame_index": len(measurements) + 1,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "capture_ms": capture_ms,
                "inference_ms": inference_ms,
                "detection_count": len(detections),
                "detections": detections,
            }
            measurements.append(row)
            score = max((item["confidence"] for item in detections), default=-1.0)
            if best is None or score > best["score"]:
                best = {"score": score, "frame": frame.copy(), "detections": detections,
                        "inference_ms": inference_ms, "frame_index": row["frame_index"]}
    finally:
        capture.release()
        resource_samples = sampler.stop()
    elapsed = time.perf_counter() - started
    if best is None:
        raise RuntimeError("No camera frames were captured; no detection screenshot was saved.")

    summary = {
        "process_cpu_percent": percentile_stats([float(row["process_cpu_percent"]) for row in resource_samples]),
        "system_cpu_percent": percentile_stats([float(row["system_cpu_percent"]) for row in resource_samples]),
        "rss_bytes": percentile_stats([float(row["rss_bytes"]) for row in resource_samples]),
        "system_memory_percent": percentile_stats([float(row["system_memory_percent"]) for row in resource_samples]),
        "resource_sample_count": len(resource_samples),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    capture_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    image = draw_detection_card(
        best["frame"], best["detections"], captured_at=capture_at, camera_device=args.camera_device,
        host=platform.node(), model_sha=model_hash, inference_ms=best["inference_ms"],
        capture_duration_s=elapsed, frame_count=len(measurements), resource_summary=summary,
    )
    detection_path = OUT_DIR / "live_detection_evidence.png"
    if not cv2.imwrite(str(detection_path), image):
        raise RuntimeError(f"Failed to write {detection_path}")

    resource_csv = OUT_DIR / "live_capture_resources.csv"
    resource_columns = [
        "timestamp_utc", "monotonic_ns", "process_cpu_percent", "system_cpu_percent",
        "per_core_cpu_percent", "rss_bytes", "vms_bytes", "system_memory_used_bytes",
        "system_memory_available_bytes", "system_memory_percent",
    ]
    with resource_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=resource_columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in resource_samples:
            flattened = dict(row)
            flattened["per_core_cpu_percent"] = json.dumps(flattened.get("per_core_cpu_percent", []))
            writer.writerow(flattened)

    metadata = {
        "captured_at_utc": capture_at,
        "capture_started_at_utc": capture_started,
        "capture_duration_seconds": elapsed,
        "host": platform.node(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "python_version": platform.python_version(),
        "openvino_version": ov.__version__,
        "device": "CPU",
        "model": {
            "name": "YOLO11n FP32 OpenVINO IR",
            "xml_path": str(xml_path),
            "xml_sha256": model_hash,
            "bin_sha256": sha256_file(bin_path),
        },
        "camera_device": args.camera_device,
        "camera_frames_captured": len(measurements),
        "warmup_inferences": 30,
        "selected_frame_index": best["frame_index"],
        "selected_frame_detections": best["detections"],
        "selected_frame_inference_ms": best["inference_ms"],
        "selected_frame_has_detection": bool(best["detections"]),
        "resource_summary_during_capture": summary,
        "resource_samples_csv": str(resource_csv.relative_to(ROOT)),
        "detection_image": str(detection_path.relative_to(ROOT)),
        "resource_plot_source_json": str(STABILITY_SOURCE.relative_to(ROOT)),
        "resource_plot_source_csv": str(RESOURCE_SOURCE.relative_to(ROOT)),
    }
    metadata_path = OUT_DIR / "live_capture_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stability = json.loads(STABILITY_SOURCE.read_text(encoding="utf-8"))
    resource_plot = render_resource_plot(stability)
    print(f"Detection image: {detection_path.relative_to(ROOT)}")
    print(f"Resource plot: {resource_plot.relative_to(ROOT)}")
    print(f"Capture metadata: {metadata_path.relative_to(ROOT)}")
    print(f"Captured frames={len(measurements)} selected detections={len(best['detections'])}; resources={len(resource_samples)} samples")
    for item in best["detections"]:
        print(f"  {item['class_id']}:{item['class_name']} confidence={item['confidence']:.3f} box="
              f"({item['x1']:.1f},{item['y1']:.1f},{item['x2']:.1f},{item['y2']:.1f})")
    print(f"Live capture resource means: CPU={summary['process_cpu_percent']['mean']:.1f}% process, "
          f"RSS={summary['rss_bytes']['mean'] / 1048576:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
