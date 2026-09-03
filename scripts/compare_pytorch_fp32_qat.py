"""Create a side-by-side comparison of the frozen native PyTorch reports.

This utility does not load a model and does not run inference.  It reads the
already completed final TEST reports produced by ``evaluate_final_test.py``
and computes a reproducible comparison for the fixed FP32 and NNCF-QAT
checkpoints.  The TEST split is therefore not re-read or re-evaluated here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FP32_REPORT = ROOT / "experiments/final_test/fp32_test_metrics.json"
DEFAULT_QAT_REPORT = ROOT / "experiments/final_test/qat_test_metrics.json"
DEFAULT_OUTPUT_DIR = ROOT / "experiments/final_test"
FP32_CHECKPOINT = ROOT / "artifacts/checkpoints/fp32/best.pt"
QAT_CHECKPOINT = ROOT / "artifacts/checkpoints/qat/qat_best.pt"
EXPECTED_CLASSES = ("ball", "gawang", "robot")


class ComparisonError(RuntimeError):
    """Raised when the frozen source reports are incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ComparisonError(f"Required frozen report does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ComparisonError(f"Expected a JSON object in {path}")
    return payload


def finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ComparisonError(f"{label} is not finite: {number!r}")
    return number


def relative_delta(fp32: float, qat: float) -> float | None:
    if fp32 == 0.0:
        return None
    return (qat - fp32) / fp32 * 100.0


def metric_row(metric: str, fp32: float, qat: float) -> dict[str, Any]:
    return {
        "metric": metric,
        "FP32 PyTorch": fp32,
        "QAT PyTorch": qat,
        "absolute_delta": qat - fp32,
        "relative_delta_percent": relative_delta(fp32, qat),
    }


def f1(precision: float, recall: float) -> float:
    total = precision + recall
    return 0.0 if total == 0.0 else 2.0 * precision * recall / total


def extract_metrics(report: dict[str, Any], label: str) -> dict[str, dict[str, float]]:
    overall_raw = report.get("overall")
    if not isinstance(overall_raw, dict):
        raise ComparisonError(f"{label} report has no overall metrics.")
    overall = {
        key: finite_number(overall_raw.get(key), f"{label} overall.{key}")
        for key in ("precision", "recall", "mAP50", "mAP50-95")
    }
    # Use the evaluator-provided F1 when present; otherwise derive it directly
    # from its final precision/recall values.
    overall["F1"] = finite_number(
        overall_raw.get("F1", f1(overall["precision"], overall["recall"])),
        f"{label} overall.F1",
    )

    per_class_raw = report.get("per_class")
    if not isinstance(per_class_raw, dict):
        raise ComparisonError(f"{label} report has no per-class metrics.")
    per_class: dict[str, dict[str, float]] = {}
    for class_name in EXPECTED_CLASSES:
        values = per_class_raw.get(class_name)
        if not isinstance(values, dict):
            raise ComparisonError(f"{label} report has no metrics for class {class_name!r}.")
        per_class[class_name] = {
            key: finite_number(values.get(key), f"{label} per_class.{class_name}.{key}")
            for key in ("precision", "recall", "mAP50", "mAP50-95")
        }
        per_class[class_name]["F1"] = finite_number(
            values.get("F1", f1(per_class[class_name]["precision"], per_class[class_name]["recall"])),
            f"{label} per_class.{class_name}.F1",
        )
    return {"overall": overall, "per_class": per_class}


def timing_row(fp32_report: dict[str, Any], qat_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fp32_timing = fp32_report.get("timing", {}).get("values", {})
    qat_timing = qat_report.get("timing", {}).get("values", {})
    fp32_latency = finite_number(fp32_timing.get("inference"), "FP32 timing.inference")
    qat_latency = finite_number(qat_timing.get("inference"), "QAT timing.inference")
    rows.append(metric_row("Latency (ms/image; diagnostic GPU)", fp32_latency, qat_latency))
    fp32_fps = 1000.0 / fp32_latency if fp32_latency > 0 else 0.0
    qat_fps = 1000.0 / qat_latency if qat_latency > 0 else 0.0
    rows.append(metric_row("FPS (derived; diagnostic GPU)", fp32_fps, qat_fps))
    return rows


def checkpoint_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label, path in (("FP32 PyTorch", FP32_CHECKPOINT), ("QAT PyTorch", QAT_CHECKPOINT)):
        item: dict[str, Any] = {"path": str(path)}
        if path.is_file():
            item["bytes"] = path.stat().st_size
            item["sha256"] = sha256_file(path)
        else:
            item["status"] = "missing; source report comparison still available"
        result[label] = item
    return result


def format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["scope", "class", "metric", "FP32 PyTorch", "QAT PyTorch", "absolute_delta", "relative_delta_percent"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Perbandingan FP32 dan QAT PyTorch",
        "",
        "## Status",
        "",
        "**DONE — laporan dibuat dari hasil evaluasi native PyTorch yang sudah dibekukan.**",
        "Tidak ada model yang dimuat dan tidak ada inference/evaluasi ulang yang dijalankan.",
        "",
        f"Split: `{payload['split']}`; images: `{payload['images']}`; annotations: `{payload['annotations']}`.",
        "",
        "## Metrik keseluruhan",
        "",
        "| Metrik | FP32 PyTorch | QAT PyTorch | Delta absolut | Delta relatif |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["overall_rows"] + payload["timing_rows"]:
        relative = row.get("relative_delta_percent")
        lines.append(
            f"| {row['metric']} | {format_value(row['FP32 PyTorch'])} | {format_value(row['QAT PyTorch'])} | "
            f"{format_value(row['absolute_delta'])} | {format_value(relative)}% |"
        )
    lines.extend(
        [
            "",
            "## Metrik per kelas",
            "",
            "| Kelas | Metrik | FP32 PyTorch | QAT PyTorch | Delta absolut | Delta relatif |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["per_class_rows"]:
        lines.append(
            f"| {row['class']} | {row['metric']} | {format_value(row['FP32 PyTorch'])} | "
            f"{format_value(row['QAT PyTorch'])} | {format_value(row['absolute_delta'])} | "
            f"{format_value(row['relative_delta_percent'])}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretasi",
            "",
            "- Precision, recall, mAP, dan F1 berasal dari evaluasi native PyTorch pada TEST yang telah dilakukan sebelumnya.",
            "- Latency/FPS adalah pengukuran diagnostik GPU dari evaluator, bukan benchmark deployment ROBOTIS OP3.",
            "- CPU utilization dan RAM tidak tersedia pada laporan ini; keduanya memerlukan benchmark terkontrol di perangkat target.",
            "- Perbedaan ini membandingkan kualitas model PyTorch FP32 dengan model PyTorch QAT, bukan validitas ekspor OpenVINO.",
            "- TEST tidak dibaca ulang, tidak dipakai untuk training, kalibrasi, tuning, atau pemilihan checkpoint dalam pembuatan laporan ini.",
            "",
            "## Artefak terkait",
            "",
            f"- FP32 report: `{payload['source_reports']['FP32 PyTorch']}`",
            f"- QAT report: `{payload['source_reports']['QAT PyTorch']}`",
            f"- JSON comparison: `{payload['outputs']['json']}`",
            f"- CSV comparison: `{payload['outputs']['csv']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(fp32_report: dict[str, Any], qat_report: dict[str, Any], fp32_path: Path, qat_path: Path) -> dict[str, Any]:
    fp32_metrics = extract_metrics(fp32_report, "FP32")
    qat_metrics = extract_metrics(qat_report, "QAT")
    fp32_images = int(fp32_report.get("test_images", 0))
    qat_images = int(qat_report.get("test_images", 0))
    fp32_annotations = int(fp32_report.get("test_annotations", 0))
    qat_annotations = int(qat_report.get("test_annotations", 0))
    if (fp32_images, fp32_annotations) != (qat_images, qat_annotations):
        raise ComparisonError("FP32 and QAT reports do not use the same image/annotation counts.")

    metric_names = ("precision", "recall", "F1", "mAP50", "mAP50-95")
    overall_rows = [metric_row(name, fp32_metrics["overall"][name], qat_metrics["overall"][name]) for name in metric_names]
    per_class_rows: list[dict[str, Any]] = []
    for class_name in EXPECTED_CLASSES:
        for name in metric_names:
            row = metric_row(name, fp32_metrics["per_class"][class_name][name], qat_metrics["per_class"][class_name][name])
            row["class"] = class_name
            per_class_rows.append(row)

    output_dir = DEFAULT_OUTPUT_DIR
    payload: dict[str, Any] = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "comparison_type": "native PyTorch FP32 vs native NNCF-QAT PyTorch",
        "split": "TEST (frozen existing reports; no re-evaluation)",
        "images": fp32_images,
        "annotations": fp32_annotations,
        "class_mapping": {"0": "ball", "1": "gawang", "2": "robot"},
        "source_reports": {"FP32 PyTorch": str(fp32_path), "QAT PyTorch": str(qat_path)},
        "checkpoint_metadata": checkpoint_metadata(),
        "overall_rows": overall_rows,
        "timing_rows": timing_row(fp32_report, qat_report),
        "per_class_rows": per_class_rows,
        "limitations": [
            "Timing is diagnostic GPU timing from the existing evaluator, not a ROBOTIS OP3 deployment benchmark.",
            "CPU utilization and RAM were not measured.",
            "This report does not assess QAT PyTorch to OpenVINO numerical equivalence.",
        ],
        "outputs": {
            "json": str(output_dir / "pytorch_comparison.json"),
            "csv": str(output_dir / "pytorch_comparison.csv"),
            "markdown": str(output_dir / "pytorch_comparison.md"),
        },
        "statement": "No training, calibration, PTQ, OpenVINO conversion, or new TEST evaluation was performed.",
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-report", type=Path, default=DEFAULT_FP32_REPORT)
    parser.add_argument("--qat-report", type=Path, default=DEFAULT_QAT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fp32_path = args.fp32_report if args.fp32_report.is_absolute() else ROOT / args.fp32_report
    qat_path = args.qat_report if args.qat_report.is_absolute() else ROOT / args.qat_report
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build_payload(read_json(fp32_path), read_json(qat_path), fp32_path, qat_path)
    payload["outputs"] = {
        "json": str(output_dir / "pytorch_comparison.json"),
        "csv": str(output_dir / "pytorch_comparison.csv"),
        "markdown": str(output_dir / "pytorch_comparison.md"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pytorch_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = []
    for row in payload["overall_rows"]:
        rows.append({"scope": "overall", "class": "all", **row})
    for row in payload["timing_rows"]:
        rows.append({"scope": "diagnostic_timing", "class": "all", **row})
    for row in payload["per_class_rows"]:
        rows.append({"scope": "per_class", **row})
    write_csv(output_dir / "pytorch_comparison.csv", rows)
    write_markdown(output_dir / "pytorch_comparison.md", payload)

    print("Native PyTorch FP32 vs QAT comparison created from frozen reports.")
    print(f"TEST images={payload['images']}, annotations={payload['annotations']}")
    for row in payload["overall_rows"]:
        print(f"{row['metric']}: FP32={row['FP32 PyTorch']:.6f}, QAT={row['QAT PyTorch']:.6f}, delta={row['absolute_delta']:.6f}")
    print(f"JSON: {output_dir / 'pytorch_comparison.json'}")
    print(f"CSV: {output_dir / 'pytorch_comparison.csv'}")
    print(f"Markdown: {output_dir / 'pytorch_comparison.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as exc:
        raise SystemExit(f"Comparison failed: {exc}") from exc
