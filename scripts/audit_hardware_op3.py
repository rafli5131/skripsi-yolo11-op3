"""Collect a non-invasive host, CPU, memory, camera, and ROS 2 audit."""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments/hardware_op3"


def command(args: list[str], timeout: float = 12.0) -> dict[str, Any]:
    executable = shutil.which(args[0])
    if not executable:
        return {"available": False, "command": args, "error": f"{args[0]} not found"}
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "command": args, "error": str(exc)}
    return {
        "available": True,
        "command": args,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def cpu_summary() -> dict[str, Any]:
    lscpu = command(["lscpu"])
    text = lscpu.get("stdout", "")
    fields = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    flags = set(fields.get("Flags", "").split())
    return {
        "model": fields.get("Model name"),
        "architecture": fields.get("Architecture", platform.machine()),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "threads_per_core": fields.get("Thread(s) per core"),
        "vendor": fields.get("Vendor ID"),
        "max_mhz": fields.get("CPU max MHz"),
        "instruction_sets": {name: name in flags for name in ("avx", "avx2", "avx512f", "avx_vnni")},
        "flags": sorted(flags),
        "lscpu": lscpu,
    }


def camera_audit() -> dict[str, Any]:
    devices = sorted(glob.glob("/dev/video*"))
    result: dict[str, Any] = {"devices": devices, "v4l2_devices": command(["v4l2-ctl", "--list-devices"])}
    if not devices:
        result["capture_test"] = {"status": "no /dev/video* device"}
        return result
    try:
        import cv2

        capture = None
        opened_device = None
        for device in devices:
            test = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if test.isOpened():
                capture, opened_device = test, device
                break
            test.release()
        if capture is None:
            result["capture_test"] = {"status": "no device opened"}
        else:
            ok, frame = capture.read()
            result["capture_test"] = {
                "status": "passed" if ok and frame is not None else "failed",
                "device": opened_device,
                "first_frame_shape": list(frame.shape) if frame is not None else None,
                "first_frame_dtype": str(frame.dtype) if frame is not None else None,
            }
            capture.release()
    except Exception as exc:  # audit should record optional camera stack failures
        result["capture_test"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return result


def openvino_audit() -> dict[str, Any]:
    try:
        import openvino as ov

        core = ov.Core()
        return {"available": True, "version": ov.__version__, "devices": core.available_devices}
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def ros_audit() -> dict[str, Any]:
    distro = os.environ.get("ROS_DISTRO")
    topics = command(["ros2", "topic", "list", "-t"], timeout=20.0)
    nodes = command(["ros2", "node", "list"], timeout=20.0)
    return {
        "distro_environment": distro,
        "version_command": command(["ros2", "--version"]),
        "topic_list": topics,
        "node_list": nodes,
        "image_topics": [
            line for line in topics.get("stdout", "").splitlines()
            if re.search(r"image|camera", line, re.IGNORECASE)
        ],
    }


def main() -> int:
    os_release = platform.freedesktop_os_release() if hasattr(platform, "freedesktop_os_release") else {}
    openvino = openvino_audit()
    ros = ros_audit()
    cameras = camera_audit()
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(ROOT)
    try:
        from openvino import __version__ as ov_version
        ov_version_for_md = ov_version
    except Exception:
        ov_version_for_md = "not available"
    try:
        import rclpy  # type: ignore[import-not-found]
        ros_python = getattr(rclpy, "__version__", "importable")
    except Exception as exc:
        ros_python = f"not importable: {type(exc).__name__}: {exc}"

    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "uname": command(["uname", "-a"]),
        "os": os_release,
        "python": {"version": sys.version, "executable": sys.executable},
        "cpu": cpu_summary(),
        "memory": {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "used_bytes": memory.used,
            "percent": memory.percent,
        },
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "lsblk": command(["lsblk", "-o", "NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS"]),
        "lsusb": command(["lsusb"]),
        "cameras": cameras,
        "openvino": openvino,
        "ros2": ros,
        "ros_python": ros_python,
        "thermal": {
            "sensors": command(["sensors"]),
            "thermal_zones": [
                {"path": str(path), "temperature_millidegrees": path.read_text().strip()}
                for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
                if path.is_file()
            ],
        },
        "host_matches_requested_nuc11": "NUC11" in " ".join(
            [platform.node(), str(cpu_summary().get("model"))]
        ).upper(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "system_info.json"
    md_path = OUT_DIR / "system_info.md"
    json_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cpu = info["cpu"]
    camera = cameras.get("capture_test", {})
    lines = [
        "# OP3 hardware and software audit",
        "",
        f"- Captured (UTC): {info['timestamp_utc']}",
        f"- Hostname: {info['hostname']}",
        f"- OS: {os_release.get('PRETTY_NAME', platform.platform())}",
        f"- Kernel: {platform.release()}",
        f"- CPU: {cpu.get('model')} ({cpu.get('physical_cores')} cores / {cpu.get('logical_cpus')} threads)",
        f"- Instruction sets: {', '.join(k for k, v in cpu['instruction_sets'].items() if v) or 'none detected'}",
        f"- RAM: {memory.total / (1024**3):.2f} GiB total; {memory.available / (1024**3):.2f} GiB available at audit",
        f"- Python: {platform.python_version()} ({sys.executable})",
        f"- OpenVINO: {ov_version_for_md}; devices: {', '.join(openvino.get('devices', [])) or 'unavailable'}",
        f"- ROS 2 distro: {ros.get('distro_environment') or 'not set'}; rclpy: {ros_python}",
        f"- Camera devices: {', '.join(cameras.get('devices', [])) or 'none'}; capture: {camera.get('status')}",
        f"- Requested NUC11 match: {info['host_matches_requested_nuc11']}",
        "",
        "Full `lscpu`, USB, block-device, ROS topic/node, camera, thermal, and resource details are in `system_info.json`.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)}")
    print(f"CPU: {cpu.get('model')}; RAM: {memory.total / (1024**3):.2f} GiB")
    print(f"OpenVINO devices: {openvino.get('devices', [])}; ROS: {ros.get('distro_environment')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
