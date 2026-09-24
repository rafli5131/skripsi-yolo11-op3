"""Evaluate four frozen models on the same VAL split using Ultralytics CPU.

This script never loads TEST, trains, exports, quantizes, or calibrates a model.
OpenVINO IRs are exposed to Ultralytics through temporary symlinks bearing
its required ``*_openvino_model`` directory suffix. Source artifacts stay put.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openvino as ov
import torch
import ultralytics
import yaml
from ultralytics import YOLO

from op3_model_audit import ptq_validation_gate, qat_validation_gate, sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "experiments/server_openvino"
DATA_YAML = ROOT / "data/dataset/data.yaml"
CLASSES = {0: "ball", 1: "gawang", 2: "robot"}
EXPECTED_IMAGES = 1213
EXPECTED_ANNOTATIONS = 1525
SPECS = (
    ("fp32_native", "FP32 native PyTorch", "pt", "artifacts/checkpoints/fp32/best.pt"),
    ("fp32_openvino", "FP32 OpenVINO", "ir", "artifacts/openvino/fp32/model.xml"),
    ("ptq_int8", "PTQ INT8 OpenVINO", "ir", "experiments/ptq/backend_models/ptq_openvino_model/model.xml"),
    ("qat_int8", "QAT INT8 OpenVINO", "ir", "artifacts/openvino/qat_int8/model.xml"),
)
EXPECTED_SHA256 = {
    "fp32_native": ("e05b2b21f7faa68f65671f0aaae37ad962aa5e25c6b8b26853fd0b389d420b8c",),
    "fp32_openvino": (
        "233bb872e218ab3bf525188b0e105eea9f2a79e60ffecb4b767e63939ba257e7",
        "e85f45438b5f7c8858b1a168a03f61b12a73b8159391529d19b84e09ff854026",
    ),
    "ptq_int8": (
        "10cb9dfa3539650a0bc46d60ad708f9ff79fc1e2dc79c5ac4a73ff4943326657",
        "82246726edcd780685657f673caa2b40719a8c22b67e3702ca3d1f226b8fcf81",
    ),
    "qat_int8": (
        "0a97a52f7a72ae9667c6f501ac1e0f44ff04bb5a860ac072e4f39ad406c424b2",
        "4a88143019b418edd879932820589c18557dfe9954196fa92025749505bc6352",
    ),
}
CONFIG = {
    "split": "val",
    "imgsz": 640,
    "batch": 1,
    "workers": 0,
    "device": "cpu",
    "cache": False,
    "fraction": 1.0,
    "augment": False,
    "amp": False,
    "conf": None,
    "iou": 0.7,
    "rect": True,
    "plots": False,
    "save_json": False,
    "save_txt": False,
}
METRICS = ("precision", "recall", "F1", "mAP50", "mAP50-95")


def require_artifacts() -> dict[str, dict[str, Any]]:
    qat = qat_validation_gate(ROOT)
    ptq = ptq_validation_gate(ROOT, ROOT / SPECS[2][3])
    if not qat["accepted"] or not ptq["accepted"]:
        raise RuntimeError(f"Provenance gate failed: QAT={qat['reasons']}, PTQ={ptq['reasons']}")

    records: dict[str, dict[str, Any]] = {}
    for slug, label, kind, relative in SPECS:
        source = ROOT / relative
        files = (source,) if kind == "pt" else (source, source.with_suffix(".bin"))
        if any(not path.is_file() for path in files):
            raise RuntimeError(f"Missing frozen artifact for {label}: {files}")
        hashes = tuple(sha256_file(path) for path in files)
        if hashes != EXPECTED_SHA256[slug]:
            raise RuntimeError(f"Frozen artifact hash changed for {label}: {hashes}")
        records[slug] = {
            "label": label,
            "format": "PyTorch checkpoint" if kind == "pt" else "OpenVINO IR",
            "files": [
                {"path": str(path.relative_to(ROOT)), "sha256": digest, "size_bytes": path.stat().st_size}
                for path, digest in zip(files, hashes)
            ],
        }
    return records


def val_dataset() -> tuple[Path, Path, int]:
    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    names = {int(key): str(value) for key, value in data["names"].items()}
    if names != CLASSES:
        raise RuntimeError(f"Unexpected class mapping: {names}")
    val = Path(data["val"]).expanduser().resolve()
    if not val.is_dir() or val.name != "images" or not val.is_relative_to(ROOT):
        raise RuntimeError(f"Invalid VAL image directory: {val}")
    labels = val.parent / "labels"
    if not labels.is_dir():
        raise RuntimeError(f"Missing VAL labels: {labels}")
    images = sorted(p for p in val.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    annotations = 0
    for image in images:
        label = labels / f"{image.stem}.txt"
        if label.is_file():
            annotations += sum(bool(line.strip()) for line in label.read_text(encoding="utf-8").splitlines())
    if len(images) != EXPECTED_IMAGES or annotations != EXPECTED_ANNOTATIONS:
        raise RuntimeError(f"VAL population changed: {len(images)} images, {annotations} annotations")
    return val, Path(data["train"]).expanduser().resolve(), annotations


def harmonic_f1(precision: float, recall: float) -> float:
    total = precision + recall
    return 0.0 if total == 0 else 2.0 * precision * recall / total


def extract_metrics(metrics: Any) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    values = [float(value) for value in metrics.mean_results()]
    if len(values) < 4:
        raise RuntimeError(f"Ultralytics returned incomplete detection metrics: {values}")
    precision, recall, map50, map5095 = values[:4]
    overall = {
        "precision": precision,
        "recall": recall,
        "F1": harmonic_f1(precision, recall),
        "mAP50": map50,
        "mAP50-95": map5095,
    }
    per_class = {
        name: {metric: 0.0 for metric in METRICS}
        for name in CLASSES.values()
    }
    for index, raw_class_id in enumerate(metrics.ap_class_index):
        class_id = int(raw_class_id)
        p, r, m50, m5095 = (float(value) for value in metrics.class_result(index)[:4])
        per_class[CLASSES[class_id]] = {
            "precision": p,
            "recall": r,
            "F1": harmonic_f1(p, r),
            "mAP50": m50,
            "mAP50-95": m5095,
        }
    return overall, per_class


def stage_ir(source_xml: Path, slug: str, temporary: Path) -> Path:
    staged = temporary / f"{slug}_openvino_model"
    staged.mkdir()
    for source in (source_xml, source_xml.with_suffix(".bin")):
        (staged / source.name).symlink_to(source)
    return staged


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_comparison(results: dict[str, dict[str, Any]], metadata: dict[str, Any]) -> None:
    names = [slug for slug, *_ in SPECS]
    baseline = results["fp32_native"]["overall"]
    comparison = {
        **metadata,
        "results": results,
        "delta_vs_fp32_native": {
            slug: {metric: results[slug]["overall"][metric] - baseline[metric] for metric in METRICS}
            for slug in names[1:]
        },
    }
    save_json(OUTPUT_DIR / "val_comparison.json", comparison)

    with (OUTPUT_DIR / "val_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["metric", *names])
        for metric in METRICS:
            writer.writerow([metric, *(results[slug]["overall"][metric] for slug in names)])
    with (OUTPUT_DIR / "val_per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["class", "metric", *names])
        for class_name in CLASSES.values():
            for metric in METRICS:
                writer.writerow([class_name, metric, *(results[slug]["per_class"][class_name][metric] for slug in names)])

    rows = [
        "# Server VAL Detection Metrics — FP32 Native vs FP32 OpenVINO vs PTQ vs QAT",
        "",
        "Evaluasi diagnostik pada host server dengan **split VAL** yang sama: 1.213 gambar dan 1.525 anotasi. Empat model memakai evaluator Ultralytics 8.4.130 pada CPU, input 640, batch 1, IoU 0.7, dan confidence default (`conf=None`). Tidak ada training, export, quantization, calibration, atau FINAL TEST dalam run ini.",
        "",
        "## Hasil keseluruhan",
        "",
        "| Metric | FP32 native | FP32 OpenVINO | PTQ INT8 | QAT INT8 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        rows.append("| " + metric + " | " + " | ".join(f"{results[slug]['overall'][metric]:.4f}" for slug in names) + " |")
    rows.extend(["", "F1 dihitung sebagai harmonic mean dari precision dan recall keseluruhan yang dilaporkan evaluator. Nilai lain berasal langsung dari evaluator; angka dibulatkan untuk tabel, sedangkan JSON/CSV menyimpan presisi aslinya.", "", "## Selisih terhadap FP32 native", "", "| Metric (poin persentase) | FP32 OpenVINO | PTQ INT8 | QAT INT8 |", "| --- | ---: | ---: | ---: |"])
    for metric in METRICS:
        rows.append("| " + metric + " | " + " | ".join(f"{100 * comparison['delta_vs_fp32_native'][slug][metric]:+.2f}" for slug in names[1:]) + " |")
    rows.extend(["", "## Per kelas", "", "| Class | Metric | FP32 native | FP32 OpenVINO | PTQ INT8 | QAT INT8 |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for class_name in CLASSES.values():
        for metric in METRICS:
            rows.append("| " + class_name + " | " + metric + " | " + " | ".join(f"{results[slug]['per_class'][class_name][metric]:.4f}" for slug in names) + " |")
    rows.extend([
        "", "## Provenance dan batasan", "",
        "Sumber model dan SHA256 ada di masing-masing `val_<model>.json`. PTQ dan QAT harus lolos provenance gate sebelum evaluasi; IR OpenVINO dijalankan melalui symlink sementara tanpa mengubah artifact aslinya. Dataset VAL canonical dibaca dari `data/dataset/data.yaml`; TEST tidak dimuat. JSON model mencatat waktu, host, commit, versi software, dan konfigurasi evaluator.",
        "", "Ini adalah kualitas deteksi **VAL pada server**, bukan accuracy FINAL TEST dan bukan benchmark latency/FPS final ROBOTIS OP3. Waktu yang dikeluarkan evaluator hanya diagnostik; bandingkan performa host memakai [benchmark server terkontrol](README.md) dan performa robot dari pengujian OP3 yang terpisah.",
        "", "Raw result: [comparison JSON](val_comparison.json), [overall CSV](val_comparison.csv), [per-class CSV](val_per_class.csv), serta `val_fp32_native.json`, `val_fp32_openvino.json`, `val_ptq_int8.json`, dan `val_qat_int8.json` di direktori ini.",
        "",
    ])
    (OUTPUT_DIR / "VAL_METRICS_REPORT.md").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    artifacts = require_artifacts()
    val_dir, train_dir, annotations = val_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata = {
        "split": "val",
        "val_images": EXPECTED_IMAGES,
        "val_annotations": annotations,
        "dataset_yaml": str(DATA_YAML.relative_to(ROOT)),
        "dataset_yaml_sha256": sha256_file(DATA_YAML),
        "val_images_path": str(val_dir),
        "evaluator_configuration": CONFIG,
        "hostname": platform.node(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "openvino_version": ov.__version__,
        "git_commit_at_evaluation": revision,
        "evaluator_script_sha256": sha256_file(Path(__file__)),
        "command": shlex.join([sys.executable, *sys.argv]),
        "accuracy_context": "VAL detection quality on server; not FINAL TEST or ROBOTIS OP3 deployment",
    }
    results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="server_val_") as temporary_name:
        temporary = Path(temporary_name)
        eval_yaml = temporary / "val_data.yaml"
        eval_yaml.write_text(yaml.safe_dump({
            "path": str(val_dir.parent.parent),
            "train": str(train_dir),
            "val": str(val_dir),
            "names": CLASSES,
            "nc": len(CLASSES),
            "task": "detect",
        }, sort_keys=False), encoding="utf-8")
        for slug, label, kind, relative in SPECS:
            source = ROOT / relative
            model_path = source if kind == "pt" else stage_ir(source, slug, temporary)
            print(f"Evaluating {label} on {EXPECTED_IMAGES} VAL images", flush=True)
            model = YOLO(str(model_path), task="detect", verbose=False)
            metrics = model.val(
                data=str(eval_yaml),
                project=str(temporary / "runs"),
                name=slug,
                exist_ok=True,
                verbose=False,
                **CONFIG,
            )
            overall, per_class = extract_metrics(metrics)
            result = {
                **metadata,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": {"slug": slug, **artifacts[slug]},
                "overall": overall,
                "per_class": per_class,
                "evaluator_return": {str(key): float(value) for key, value in (metrics.results_dict or {}).items()},
                "evaluator_speed_ms_per_image_diagnostic_only": {str(key): float(value) for key, value in metrics.speed.items()},
            }
            results[slug] = result
            save_json(OUTPUT_DIR / f"val_{slug}.json", result)
            print(f"{slug}: P={overall['precision']:.4f} R={overall['recall']:.4f} F1={overall['F1']:.4f} mAP50={overall['mAP50']:.4f} mAP50-95={overall['mAP50-95']:.4f}", flush=True)
    save_comparison(results, metadata)
    print("Wrote VAL model JSONs, comparison JSON/CSVs, and VAL_METRICS_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
