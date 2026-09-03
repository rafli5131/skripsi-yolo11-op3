"""Build and evaluate an isolated FP32 -> PTQ INT8 OpenVINO comparator.

This script is deliberately separate from the true-NNCF-QAT scripts.  It
starts from the fixed FP32 checkpoint, calibrates only on a deterministic
subset of TRAIN images, and uses the installed Ultralytics
``torch2openvino(..., quantize=8)`` path.  In Ultralytics 8.4.130 that path
calls ``nncf.quantize`` on the converted OpenVINO model; this is PTQ and is
not used by the QAT training or QAT restore paths.

No QAT checkpoint, TEST image, or training is used here.  The historical QAT
OpenVINO IR is normally inventoried as ``REJECTED / diagnostic only``.  With
the explicit ``--evaluate-rejected-qat`` option it is executed on VAL solely
to display diagnostic detection metrics; it is never promoted or treated as a
valid deployment artifact.  The generated PTQ model is an independent
comparator for the research metrics and never overwrites the FP32 or QAT
artifacts.

Run from the project root on the SSH host (or another environment that has
the project dependencies).  Conversion and validation use CPU so this
experiment does not require a visible CUDA device.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path.cwd().resolve()
load_dotenv(ROOT / ".env", override=False)

import cv2
import nncf
import numpy as np
import openvino as ov
import torch
import torchvision
import ultralytics
import yaml
from nncf import Dataset as NNCFDataset
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.export.openvino import torch2openvino

from probe_qat_compatibility import attach_ultralytics_training_args
from export_openvino import enable_ultralytics_detect_export


FP32_CHECKPOINT = Path("artifacts/checkpoints/fp32/best.pt")
FP32_OPENVINO_XML = Path("artifacts/openvino/fp32/model.xml")
PTQ_OPENVINO_DIR = Path("artifacts/openvino/ptq_int8")
QAT_OPENVINO_XML = Path("artifacts/openvino/qat_int8/model.xml")
EXPERIMENT_DIR = Path("experiments/ptq")
MANIFEST_PATH = EXPERIMENT_DIR / "manifest.json"
PTQ_METRICS_PATH = EXPERIMENT_DIR / "ptq_val_metrics.json"
FP32_METRICS_PATH = EXPERIMENT_DIR / "fp32_openvino_val_metrics.json"
COMPARISON_PATH = EXPERIMENT_DIR / "comparison.json"
COMPARISON_CSV_PATH = EXPERIMENT_DIR / "comparison.csv"
COMPARISON_MD_PATH = EXPERIMENT_DIR / "comparison.md"
EVAL_DATA_PATH = EXPERIMENT_DIR / "data_train_val.yaml"
DATA_YAML = Path("data/dataset/data.yaml")
BASE_CONFIG = Path("configs/base.yaml")
EXPECTED_NAMES = {0: "ball", 1: "gawang", 2: "robot"}
IMGSZ = 640
DEFAULT_CALIBRATION_SAMPLES = 300
DEFAULT_VAL_FRACTION = 1.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PTQStop(RuntimeError):
    """Controlled PTQ failure recorded in the manifest."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help=(
            "Number of deterministic TRAIN images used for PTQ calibration. "
            f"Default: {DEFAULT_CALIBRATION_SAMPLES} (the installed Ultralytics exporter recommends >=300)."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="Fraction of VAL used for diagnostic metrics (default: 1.0; TEST is never loaded).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing only the existing PTQ output directory and PTQ experiment reports.",
    )
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help=(
            "Skip PTQ conversion and evaluate the existing artifacts/openvino/ptq_int8 IR. "
            "Use this after a successful export when only the comparison/evaluator needs to be rerun."
        ),
    )
    parser.add_argument(
        "--evaluate-rejected-qat",
        action="store_true",
        help=(
            "Also evaluate the existing rejected artifacts/openvino/qat_int8 IR on VAL for diagnostic display only. "
            "It is never treated as a valid QAT deployment result."
        ),
    )
    return parser.parse_args()


def require_project_root() -> None:
    if not (ROOT / "pyproject.toml").is_file():
        raise PTQStop("Run this script from the project root containing pyproject.toml.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def normalized_names(value: Any) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(key): str(name) for key, name in value.items()}
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    raise PTQStop(f"Unsupported class mapping type: {type(value).__name__}")


def load_dataset_config() -> tuple[dict[str, Any], Path, Path, Any]:
    data_path = (ROOT / DATA_YAML).resolve()
    if not data_path.is_file():
        raise PTQStop(f"Canonical dataset YAML not found: {data_path}")
    payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PTQStop(f"Invalid canonical dataset YAML: {data_path}")
    names = normalized_names(payload.get("names"))
    if names != EXPECTED_NAMES:
        raise PTQStop(f"Class mapping changed: expected {EXPECTED_NAMES}, got {names}")

    def resolve_split(split: str) -> Path:
        value = payload.get(split)
        if not isinstance(value, str) or not value:
            raise PTQStop(f"Canonical dataset YAML has no valid {split!r} path.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            root_value = payload.get("path")
            dataset_root = Path(root_value).expanduser() if isinstance(root_value, str) and root_value else data_path.parent
            if not dataset_root.is_absolute():
                dataset_root = (data_path.parent / dataset_root).resolve()
            path = dataset_root / path
        path = path.resolve()
        if not path.is_dir() or path.name != "images" or not (path.parent / "labels").is_dir():
            raise PTQStop(f"Resolved {split} path is not an images directory with sibling labels: {path}")
        if not path.is_relative_to(ROOT):
            raise PTQStop(f"Resolved {split} path is outside the current project root: {path}")
        return path

    # Deliberately resolve only train and val.  The test entry is not read,
    # enumerated, cached, or passed to any evaluator in this script.
    train_images = resolve_split("train")
    val_images = resolve_split("val")
    base = yaml.safe_load((ROOT / BASE_CONFIG).read_text(encoding="utf-8"))
    version = base.get("dataset", {}).get("version") if isinstance(base, dict) else None
    data = {
        "train": str(train_images),
        "val": str(val_images),
        "names": EXPECTED_NAMES,
        "nc": len(EXPECTED_NAMES),
        "channels": 3,
    }
    return data, train_images, val_images, version


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise PTQStop(f"No supported images found in {directory}")
    return paths


def preprocess_cpu(image_path: Path) -> np.ndarray:
    """Match the deployment preprocessing used by the previous diagnostics."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise PTQStop(f"Unable to decode calibration/evaluation image: {image_path}")
    image = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False, stride=32)(image=image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor[None, ...]


def make_calibration_dataset(train_dir: Path, count: int) -> tuple[NNCFDataset, list[Path]]:
    paths = image_paths(train_dir)
    if count < 1:
        raise PTQStop("--calibration-samples must be positive.")
    if count > len(paths):
        raise PTQStop(f"Requested {count} TRAIN calibration images but only {len(paths)} exist.")
    selected = paths[:count]
    # The iterable contains paths, so images are decoded lazily on every NNCF
    # calibration pass instead of retaining hundreds of 640x640 tensors.
    dataset = NNCFDataset(selected, transform_func=preprocess_cpu)
    return dataset, selected


def write_eval_yaml(data: dict[str, Any]) -> Path:
    payload = {
        "path": str(ROOT / "data" / "raw" / "advance_vision_human"),
        "train": data["train"],
        "val": data["val"],
        "names": EXPECTED_NAMES,
        "nc": len(EXPECTED_NAMES),
        "task": "detect",
    }
    write_json(EVAL_DATA_PATH.with_suffix(".json"), payload)
    EVAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_DATA_PATH.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return EVAL_DATA_PATH


def load_fp32_cpu(path: Path) -> torch.nn.Module:
    model = YOLO(str(path), task="detect").model
    attach_ultralytics_training_args(model, ROOT)
    if normalized_names(model.names) != EXPECTED_NAMES:
        raise PTQStop(f"FP32 checkpoint class mapping differs from {EXPECTED_NAMES}.")
    return model.cpu().float().eval()


def graph_report(xml_path: Path) -> dict[str, Any]:
    model = ov.Core().read_model(str(xml_path), weights=xml_path.with_suffix(".bin"))
    operations = model.get_ordered_ops()
    op_counts = Counter(operation.get_type_name() for operation in operations)
    constant_types: Counter[str] = Counter()
    for operation in operations:
        if operation.get_type_name() != "Constant":
            continue
        for index in range(operation.get_output_size()):
            constant_types[str(operation.get_output_element_type(index))] += 1
    return {
        "input shapes": [str(port.get_partial_shape()) for port in model.inputs],
        "output shapes": [str(port.get_partial_shape()) for port in model.outputs],
        "input element types": [str(port.get_element_type()) for port in model.inputs],
        "output element types": [str(port.get_element_type()) for port in model.outputs],
        "total operation count": len(operations),
        "operation counts": dict(sorted(op_counts.items())),
        "FakeQuantize count": op_counts.get("FakeQuantize", 0),
        "Convert count": op_counts.get("Convert", 0),
        "constant element-type counts": dict(sorted(constant_types.items())),
        "quantization semantics visible": bool(op_counts.get("FakeQuantize", 0)),
    }


def file_report(xml_path: Path) -> dict[str, Any]:
    bin_path = xml_path.with_suffix(".bin")
    if not xml_path.is_file() or not bin_path.is_file():
        raise PTQStop(f"Expected PTQ IR pair was not produced: {xml_path} and {bin_path}")
    return {
        "XML": str(xml_path),
        "BIN": str(bin_path),
        "XML bytes": xml_path.stat().st_size,
        "BIN bytes": bin_path.stat().st_size,
        "total bytes": xml_path.stat().st_size + bin_path.stat().st_size,
        "XML SHA256": sha256_file(xml_path),
        "BIN SHA256": sha256_file(bin_path),
    }


def count_annotations(image_dir: Path, selected_fraction: float = 1.0) -> tuple[int, int]:
    paths = image_paths(image_dir)
    selected_count = len(paths) if selected_fraction >= 1.0 else max(1, int(len(paths) * selected_fraction))
    selected = paths[:selected_count]
    annotations = 0
    for image_path in selected:
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if label_path.is_file():
            annotations += sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return selected_count, annotations


def extract_metrics(metrics: Any, label: str, val_fraction: float, val_dir: Path) -> dict[str, Any]:
    try:
        mean = [float(value) for value in metrics.mean_results()]
    except Exception as error:
        raise PTQStop(f"Could not extract detection metrics for {label}: {error}") from error
    if len(mean) < 4:
        raise PTQStop(f"Ultralytics evaluator returned fewer than four detection metrics for {label}: {mean}")
    precision, recall, map50, map5095 = mean[:4]
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    per_class = {
        name: {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0, "F1": 0.0}
        for name in EXPECTED_NAMES.values()
    }
    for index, class_id_raw in enumerate(metrics.ap_class_index):
        class_id = int(class_id_raw)
        values = [float(value) for value in metrics.class_result(index)]
        p, r, m50, m5095 = values[:4]
        denom = p + r
        per_class[EXPECTED_NAMES[class_id]] = {
            "precision": p,
            "recall": r,
            "mAP50": m50,
            "mAP50-95": m5095,
            "F1": 0.0 if denom == 0 else 2.0 * p * r / denom,
        }
    val_images, val_annotations = count_annotations(val_dir, val_fraction)
    return {
        "model": label,
        "overall": {
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50-95": map5095,
            "F1": f1,
        },
        "per_class": per_class,
        "val_images": val_images,
        "val_annotations": val_annotations,
        "evaluator_return": {str(key): float(value) for key, value in (metrics.results_dict or {}).items()},
        "speed_ms_per_image": {str(key): float(value) for key, value in metrics.speed.items()},
        "speed_label": "Diagnostic host-CPU validation timing; not the ROBOTIS OP3 deployment benchmark.",
        "evaluator_configuration": {
            "split": "val",
            "imgsz": IMGSZ,
            "batch": 1,
            "workers": 0,
            "device": "cpu",
            "cache": False,
            "fraction": val_fraction,
            "augment": False,
            "conf": "Ultralytics validation default from conf=None; not tuned",
            "iou": 0.7,
            "plots": False,
        },
    }


def stage_openvino_model(xml_path: Path, slug: str) -> Path:
    """Stage an IR pair under Ultralytics' required ``*_openvino_model`` name.

    AutoBackend 8.4.130 detects OpenVINO from the path/name convention rather
    than from a bare ``model.xml`` suffix.  The source IR is copied (never
    modified) to a disposable experiment directory so ``YOLO.val`` can use
    the installed backend without changing the published artifact.
    """
    if not xml_path.is_file():
        raise PTQStop(f"OpenVINO model not found: {xml_path}")
    bin_path = xml_path.with_suffix(".bin")
    if not bin_path.is_file():
        raise PTQStop(f"OpenVINO weights not found beside {xml_path}: {bin_path}")
    # Use an absolute project-local path.  A relative ``project``/staging
    # path can be interpreted by Ultralytics relative to its historical
    # ``runs_dir`` (which may point outside this checkout on a moved project).
    staged_dir = ROOT / EXPERIMENT_DIR / "backend_models" / f"{slug}_openvino_model"
    staged_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xml_path, staged_dir / "model.xml")
    shutil.copy2(bin_path, staged_dir / "model.bin")
    return staged_dir


def evaluate_openvino(xml_path: Path, label: str, eval_yaml: Path, val_fraction: float, output_name: str, slug: str) -> dict[str, Any]:
    if not xml_path.is_file():
        raise PTQStop(f"OpenVINO model not found for {label}: {xml_path}")
    staged_dir = stage_openvino_model(xml_path, slug)
    # Ultralytics 8.4.130 requires a path containing ``_openvino_model`` for
    # format detection.  The actual XML/BIN artifact remains at xml_path.
    model = YOLO(str(staged_dir), task="detect", verbose=False)
    metrics = model.val(
        data=str(eval_yaml),
        split="val",
        imgsz=IMGSZ,
        batch=1,
        workers=0,
        device="cpu",
        cache=False,
        fraction=val_fraction,
        augment=False,
        amp=False,
        conf=None,
        iou=0.7,
        plots=False,
        save_json=False,
        save_txt=False,
        # Ultralytics resolves relative project names below its configured
        # runs_dir.  Absolute path is required here so the diagnostic run
        # cannot attempt to write to the old read-only ``skripsi`` checkout.
        project=str((ROOT / EXPERIMENT_DIR).resolve()),
        name=output_name,
        exist_ok=True,
        verbose=True,
    )
    val_dir = Path(yaml.safe_load(eval_yaml.read_text(encoding="utf-8"))["val"])
    report = extract_metrics(metrics, label, val_fraction, val_dir)
    report["source IR"] = str(xml_path)
    report["staged backend path"] = str(staged_dir)
    report["file report"] = file_report(xml_path)
    speed_values = report.get("speed_ms_per_image", {})
    latency_ms = sum(float(value) for value in speed_values.values()) if speed_values else None
    report["diagnostic latency_ms_per_image"] = latency_ms
    report["diagnostic FPS"] = None if not latency_ms or latency_ms <= 0 else 1000.0 / latency_ms
    report["CPU utilization"] = "NOT MEASURED — requires controlled ROBOTIS OP3 benchmark"
    report["RAM"] = "NOT MEASURED — requires controlled ROBOTIS OP3 benchmark"
    return report


def comparison_rows(fp32: dict[str, Any], ptq: dict[str, Any], qat: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in ("precision", "recall", "mAP50", "mAP50-95", "F1"):
        baseline = float(fp32["overall"][metric])
        optimized = float(ptq["overall"][metric])
        delta = optimized - baseline
        rows.append(
            {
                "scope": "overall",
                "class": "all",
                "metric": metric,
                "FP32 OpenVINO": baseline,
                "PTQ INT8": optimized,
                "QAT INT8": None if qat is None else float(qat["overall"][metric]),
                "absolute_delta": delta,
                "relative_delta_percent": None if baseline == 0 else delta / baseline * 100.0,
            }
        )
    for class_name in EXPECTED_NAMES.values():
        for metric in ("precision", "recall", "mAP50", "mAP50-95", "F1"):
            baseline = float(fp32["per_class"][class_name][metric])
            optimized = float(ptq["per_class"][class_name][metric])
            delta = optimized - baseline
            rows.append(
                {
                "scope": "per_class",
                "class": class_name,
                "metric": metric,
                "FP32 OpenVINO": baseline,
                "PTQ INT8": optimized,
                "QAT INT8": None if qat is None else float(qat["per_class"][class_name][metric]),
                "absolute_delta": delta,
                "relative_delta_percent": None if baseline == 0 else delta / baseline * 100.0,
            }
        )
    return rows


def resource_and_size_rows(fp32: dict[str, Any], ptq: dict[str, Any], qat: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Build the non-accuracy rows requested for the final comparison table."""
    reports = {"FP32 OpenVINO": fp32, "PTQ INT8": ptq, "QAT INT8": qat}
    values: dict[str, dict[str, Any]] = {}
    for metric, key in (
        ("Latency", "diagnostic latency_ms_per_image"),
        ("FPS", "diagnostic FPS"),
        ("CPU utilization", "CPU utilization"),
        ("RAM", "RAM"),
    ):
        values[metric] = {name: None if report is None else report.get(key) for name, report in reports.items()}
    values["Ukuran model"] = {
        name: None if report is None else report.get("file report", {}).get("total bytes") for name, report in reports.items()
    }
    return [
        {
            "scope": "overall",
            "class": "all",
            "metric": metric,
            "FP32 OpenVINO": metric_values["FP32 OpenVINO"],
            "PTQ INT8": metric_values["PTQ INT8"],
            "QAT INT8": metric_values["QAT INT8"],
            "absolute_delta": None,
            "relative_delta_percent": None,
        }
        for metric, metric_values in values.items()
    ]


def format_cell(value: Any, metric: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if metric == "Ukuran model":
            return f"{int(value):,} B"
        if metric in {"Latency", "FPS"}:
            return f"{float(value):.3f}"
        return f"{float(value):.6f}"
    return str(value)


def format_delta(value: Any, metric: str = "") -> str:
    if value is None:
        return "—"
    return format_cell(value, metric)


def write_comparison_markdown(path: Path, accuracy_rows: list[dict[str, Any]], resource_rows: list[dict[str, Any]], qat_status: str) -> None:
    lines = [
        "# Perbandingan Akhir OpenVINO (VAL)",
        "",
        "> Metrik akurasi dan timing di bawah adalah hasil evaluasi diagnostik pada VAL. CPU utilization/RAM final harus diukur pada ROBOTIS OP3; timing host bukan benchmark deployment akhir.",
        "",
        f"Status QAT OpenVINO: **{qat_status}**",
        "",
        "| Metrik | FP32 OpenVINO | PTQ INT8 | QAT INT8 |",
        "|---|---:|---:|---:|",
    ]
    for row in accuracy_rows[:5]:
        lines.append(
            f"| {row['metric']} | {format_cell(row['FP32 OpenVINO'], row['metric'])} | {format_cell(row['PTQ INT8'], row['metric'])} | {format_cell(row['QAT INT8'], row['metric'])} |"
        )
    for row in resource_rows:
        lines.append(
            f"| {row['metric']} | {format_cell(row['FP32 OpenVINO'], row['metric'])} | {format_cell(row['PTQ INT8'], row['metric'])} | {format_cell(row['QAT INT8'], row['metric'])} |"
        )
    qat_note = (
        "Nilai QAT pada tabel adalah diagnostik dari IR yang sebelumnya ditolak; jangan gunakan sebagai klaim deployment."
        if any(row.get("QAT INT8") is not None for row in accuracy_rows)
        else "Artefak QAT berstatus REJECTED/UNRESOLVED dan ditampilkan sebagai `—`; tidak boleh dianggap sebagai hasil deployment QAT yang valid."
    )
    lines.extend(
        [
            "",
            f"Catatan: QAT INT8 hanya boleh dipakai sebagai hasil final jika artefak QAT OpenVINO telah lulus verifikasi numerik dan graph quantization. {qat_note}",
            "",
            "## Delta PTQ terhadap FP32",
            "",
            "| Metrik | Delta absolut | Delta relatif |",
            "|---|---:|---:|",
        ]
    )
    for row in accuracy_rows[:5]:
        lines.append(
            f"| {row['metric']} | {format_delta(row['absolute_delta'], row['metric'])} | {format_delta(row['relative_delta_percent'], row['metric'])}{'' if row['relative_delta_percent'] is None else '%'} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "method": "FP32 checkpoint -> Ultralytics torch2openvino(quantize=8) -> OpenVINO NNCF PTQ",
        "QAT checkpoint used": False,
        "QAT training": False,
        "TEST split loaded": False,
        "PTQ only": True,
    }
    try:
        require_project_root()
        if args.calibration_samples < 1:
            raise PTQStop("--calibration-samples must be positive.")
        if not 0.0 < args.val_fraction <= 1.0:
            raise PTQStop("--val-fraction must be in (0, 1].")
        fp32_path = (ROOT / FP32_CHECKPOINT).resolve()
        fp32_ov_path = (ROOT / FP32_OPENVINO_XML).resolve()
        if not fp32_path.is_file():
            raise PTQStop(f"Fixed FP32 checkpoint is missing: {fp32_path}")
        if not fp32_ov_path.is_file():
            raise PTQStop(f"Accepted FP32 OpenVINO baseline is missing: {fp32_ov_path}")
        if PTQ_OPENVINO_DIR.exists() and not args.overwrite and not args.evaluate_existing:
            raise PTQStop(
                f"PTQ output already exists at {ROOT / PTQ_OPENVINO_DIR}. Use --overwrite to replace only PTQ outputs."
            )
        PTQ_OPENVINO_DIR.mkdir(parents=True, exist_ok=True)
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

        data, train_dir, val_dir, dataset_version = load_dataset_config()
        calibration_dataset = None
        calibration_paths: list[Path] = []
        if not args.evaluate_existing:
            calibration_dataset, calibration_paths = make_calibration_dataset(train_dir, args.calibration_samples)
        else:
            # Preserve the deterministic calibration provenance in the
            # manifest without decoding/calibrating again.  The existing IR
            # was produced by the preceding full run (300 TRAIN images in the
            # recorded host execution); this branch only reconstructs the
            # filename list for metadata.
            train_paths = image_paths(train_dir)
            if args.calibration_samples > len(train_paths):
                raise PTQStop(
                    f"Recorded calibration count {args.calibration_samples} exceeds available TRAIN images ({len(train_paths)})."
                )
            calibration_paths = train_paths[: args.calibration_samples]
        existing_ptq_xml = ROOT / PTQ_OPENVINO_DIR / "model.xml"
        if args.evaluate_existing and not existing_ptq_xml.is_file():
            raise PTQStop(f"--evaluate-existing requested but PTQ IR is missing: {existing_ptq_xml}")
        eval_yaml = write_eval_yaml(data)
        manifest.update(
            {
                "source FP32 checkpoint": str(fp32_path),
                "source FP32 SHA256": sha256_file(fp32_path),
                "source FP32 OpenVINO": str(fp32_ov_path),
                "source FP32 OpenVINO SHA256": sha256_file(fp32_ov_path),
                "dataset version": dataset_version,
                "dataset YAML": str((ROOT / DATA_YAML).resolve()),
                "resolved TRAIN images": str(train_dir),
                "resolved VAL images": str(val_dir),
                "class mapping": EXPECTED_NAMES,
                "calibration": {
                    "split": "TRAIN only",
                    "sample count": len(calibration_paths),
                    "selection": "lexicographically first image paths",
                    "filenames": [path.name for path in calibration_paths],
                    "preprocessing": "cv2 decode -> LetterBox(640, auto=False, stride=32) -> BGR to RGB -> NCHW float32 / 255",
                },
                "validation": {
                    "split": "VAL only",
                    "fraction": args.val_fraction,
                    "test": "not loaded or used",
                },
                "versions": {
                    "Python": platform.python_version(),
                    "torch": torch.__version__,
                    "torch CUDA": torch.version.cuda,
                    "torchvision": torchvision.__version__,
                    "ultralytics": ultralytics.__version__,
                    "nncf": nncf.__version__,
                    "openvino": ov.__version__,
                },
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "execution device": "CPU for conversion and validation; no CUDA device is required",
                "input shape": [1, 3, IMGSZ, IMGSZ],
                "PTQ API": "ultralytics.utils.export.openvino.torch2openvino(..., quantize=8, calibration_dataset=nncf.Dataset(...))",
                "torch2openvino signature": str(inspect.signature(torch2openvino)),
                "underlying NNCF API": "nncf.quantize (PTQ only; never used by QAT scripts)",
                "calibration occurred": not args.evaluate_existing,
                "calibration performed in current run": not args.evaluate_existing,
                "PTQ IR reused": bool(args.evaluate_existing),
                "QAT export": "not attempted",
                "evaluation-only": bool(args.evaluate_existing),
            }
        )
        write_json(MANIFEST_PATH, manifest)
        print(f"FP32 source: {fp32_path}")
        print(
            "PTQ calibration: skipped (reusing existing IR)"
            if args.evaluate_existing
            else f"PTQ calibration: {len(calibration_paths)} TRAIN images"
        )
        print(f"VAL evaluation fraction: {args.val_fraction}")
        print("TEST: not loaded")
        print("PTQ API: torch2openvino(quantize=8) -> NNCF nncf.quantize on OpenVINO graph")

        if args.evaluate_existing:
            ptq_xml = existing_ptq_xml
            print(f"Reusing existing PTQ OpenVINO model: {ptq_xml}")
        else:
            model = load_fp32_cpu(fp32_path)
            enable_ultralytics_detect_export(model)
            example = torch.from_numpy(preprocess_cpu(calibration_paths[0])).float()
            with torch.inference_mode():
                exported_output = model(example)
            if isinstance(exported_output, tuple):
                exported_output = exported_output[0]
            if not isinstance(exported_output, torch.Tensor) or not torch.isfinite(exported_output).all():
                raise PTQStop("FP32 export model produced a non-finite or non-tensor output before PTQ export.")
            print(f"FP32 export forward shape: {list(exported_output.shape)}")

            # This is the installed Ultralytics PTQ route.  It internally calls
            # nncf.quantize only because quantize=8 was explicitly requested here.
            torch2openvino(
                model=model,
                im=example,
                output_dir=ROOT / PTQ_OPENVINO_DIR,
                dynamic=False,
                quantize=8,
                calibration_dataset=calibration_dataset,
                int8_detect=True,
                prefix="PTQ INT8:",
            )
            ptq_xml = ROOT / PTQ_OPENVINO_DIR / "model.xml"
        ptq_files = file_report(ptq_xml)
        ptq_graph = graph_report(ptq_xml)
        if not ptq_graph["quantization semantics visible"]:
            raise PTQStop("PTQ export produced an IR without visible FakeQuantize operations.")
        manifest["PTQ artifact"] = {"files": ptq_files, "graph": ptq_graph}
        write_json(MANIFEST_PATH, manifest)
        print(f"PTQ OpenVINO model: {ptq_xml}")
        print(
            f"PTQ graph: ops={ptq_graph['total operation count']}, "
            f"FakeQuantize={ptq_graph['FakeQuantize count']}, Convert={ptq_graph['Convert count']}"
        )

        fp32_report = evaluate_openvino(
            fp32_ov_path, "FP32 OpenVINO", eval_yaml, args.val_fraction, "fp32_openvino_val", "fp32"
        )
        ptq_report = evaluate_openvino(
            ptq_xml, "PTQ INT8", eval_yaml, args.val_fraction, "ptq_int8_val", "ptq"
        )

        # The currently available QAT IR is a historical/rejected artifact
        # (its PyTorch-to-IR raw-output equivalence did not pass).  It may be
        # evaluated on VAL for diagnostic comparison only, but it is never
        # promoted or described as a valid deployment model here.
        qat_report: dict[str, Any] | None = None
        qat_status = "UNAVAILABLE — no QAT OpenVINO IR found"
        qat_xml = ROOT / QAT_OPENVINO_XML
        if qat_xml.is_file() and qat_xml.with_suffix(".bin").is_file():
            qat_status = "REJECTED / diagnostic only — prior QAT OpenVINO equivalence failure"
            if args.evaluate_rejected_qat:
                # Explicitly requested by the user: run the rejected IR on
                # VAL so its observed detector metrics are visible.  The
                # report and table retain the rejection label.
                qat_report = evaluate_openvino(
                    qat_xml,
                    "QAT INT8 (REJECTED diagnostic)",
                    eval_yaml,
                    args.val_fraction,
                    "qat_int8_rejected_val",
                    "qat",
                )
                qat_report["status"] = "REJECTED / diagnostic only"
                qat_report["reason"] = "Existing QAT IR failed the prior numerical equivalence gate; do not deploy."
                write_json(EXPERIMENT_DIR / "qat_rejected_openvino_val_metrics.json", qat_report)
                qat_status = "REJECTED / diagnostic metrics displayed — not valid for deployment"
            else:
                # Its size is retained for inventory, while QAT metric and
                # timing cells remain unavailable unless the explicit flag is
                # supplied.
                qat_report = {
                    "status": "REJECTED / diagnostic only",
                    "reason": "Existing QAT IR failed the prior numerical equivalence gate; do not deploy.",
                    "source IR": str(qat_xml),
                    "file report": file_report(qat_xml),
                }
        write_json(FP32_METRICS_PATH, fp32_report)
        write_json(PTQ_METRICS_PATH, ptq_report)
        # Never present the rejected QAT IR's accuracy/timing as a final
        # deployment result.  Keep its file size visible for inventory, while
        # the QAT metric cells remain an honest dash in the final table.
        qat_comparison_report = qat_report if args.evaluate_rejected_qat else None
        # When the explicit diagnostic flag is used, retain the measured
        # latency/FPS too.  These values are still diagnostic-only because
        # the QAT IR is rejected; they are not deployment claims.
        qat_size_report = (
            qat_report
            if args.evaluate_rejected_qat
            else None if qat_report is None else {"file report": qat_report.get("file report")}
        )
        if qat_report is not None and "file report" not in qat_report:
            qat_report["file report"] = file_report(ROOT / QAT_OPENVINO_XML)
        rows = comparison_rows(fp32_report, ptq_report, qat_comparison_report)
        resource_rows = resource_and_size_rows(fp32_report, ptq_report, qat_size_report)
        comparison = {
            "models": {
                "FP32 OpenVINO": {"status": "VALID", "metrics": str(FP32_METRICS_PATH)},
                "PTQ INT8": {"status": "VALID comparator", "metrics": str(PTQ_METRICS_PATH)},
                "QAT INT8": {
                    "status": qat_status,
                    "metrics": None
                    if not args.evaluate_rejected_qat
                    else str(EXPERIMENT_DIR / "qat_rejected_openvino_val_metrics.json"),
                },
            },
            "QAT rejected diagnostic metrics": None
            if not args.evaluate_rejected_qat
            else str(EXPERIMENT_DIR / "qat_rejected_openvino_val_metrics.json"),
            "accuracy_rows": rows,
            "resource_rows": resource_rows,
            "rows": rows + resource_rows,
            "interpretation": (
                "PTQ is an independent comparator. QAT INT8 values, when present, are diagnostic only because the existing QAT IR was rejected; they must not be used as a valid deployment claim."
            ),
            "evaluation_split": "VAL only; TEST was not loaded",
        }
        write_json(COMPARISON_PATH, comparison)
        with COMPARISON_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
            fields = (
                "scope",
                "class",
                "metric",
                "FP32 OpenVINO",
                "PTQ INT8",
                "QAT INT8",
                "absolute_delta",
                "relative_delta_percent",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows + resource_rows)
        write_comparison_markdown(COMPARISON_MD_PATH, rows, resource_rows, qat_status)

        manifest.update(
            {
                "FP32 OpenVINO metrics": str(FP32_METRICS_PATH),
                "PTQ OpenVINO metrics": str(PTQ_METRICS_PATH),
                "QAT OpenVINO metrics": None
                if not args.evaluate_rejected_qat
                else str(EXPERIMENT_DIR / "qat_rejected_openvino_val_metrics.json"),
                "QAT OpenVINO status": qat_status,
                "comparison JSON": str(COMPARISON_PATH),
                "comparison CSV": str(COMPARISON_CSV_PATH),
                "comparison Markdown": str(COMPARISON_MD_PATH),
                "status": "passed",
            }
        )
        print("\nOpenVINO comparison (VAL):")
        print("metric        FP32 OV    PTQ INT8     QAT INT8")
        for row in rows[:5]:
            print(
                f"{row['metric']:<12} {format_cell(row['FP32 OpenVINO']):>10}  "
                f"{format_cell(row['PTQ INT8']):>10}  {format_cell(row['QAT INT8']):>10}"
            )
        for row in resource_rows:
            print(
                f"{row['metric']:<12} {format_cell(row['FP32 OpenVINO'], row['metric']):>10}  "
                f"{format_cell(row['PTQ INT8'], row['metric']):>10}  {format_cell(row['QAT INT8'], row['metric']):>10}"
            )
        print(f"QAT OpenVINO status: {qat_status}")
        print("PTQ OpenVINO export and VAL comparison passed.")
    except Exception as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        print(f"PTQ OpenVINO experiment failed: {manifest['error']}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        write_json(MANIFEST_PATH, manifest)
        print(f"Manifest: {MANIFEST_PATH}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
