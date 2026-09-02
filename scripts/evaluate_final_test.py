"""Evaluate the fixed final FP32 and NNCF-QAT PyTorch models on held-out TEST.

This is quality evaluation only.  TEST is read-only and is never used for
training, quantizer initialization, checkpoint selection, or tuning.  Any
reported PyTorch/CUDA timing is diagnostic, not a deployment benchmark.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

ROOT = Path.cwd().resolve()
load_dotenv(ROOT / ".env", override=False)

import nncf
import torch
import torchvision
import ultralytics
import yaml
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, load_state
from torch.utils.data import Dataset
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.data.build import build_dataloader
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.val import DetectionValidator

from probe_qat_compatibility import (
    attach_ultralytics_training_args,
    require_cuda,
    require_project_root,
    restore_ultralytics_trainer_trainability,
)
from smoke_train_qat import install_quantizer_counter, quantizers, quantizers_enabled, sha256_file


DATA_YAML = Path("data/dataset/data.yaml")
BASE_CONFIG = Path("configs/base.yaml")
FP32_CHECKPOINT = Path("artifacts/checkpoints/fp32/best.pt")
QAT_CHECKPOINT = Path("artifacts/checkpoints/qat/qat_best.pt")
OUTPUT_DIR = Path("experiments/final_test")
EXPECTED_NAMES = {0: "ball", 1: "gawang", 2: "robot"}
EXPECTED_QUANTIZERS = 184
IMGSZ = 640
BATCH = 16
WORKERS = 2
TEST_ONLY_STATEMENT = (
    "Test split was used for final evaluation only and was not used for training, quantizer initialization, "
    "checkpoint selection, or hyperparameter tuning."
)
TIMING_LABEL = "DIAGNOSTIC — NOT FINAL DEPLOYMENT BENCHMARK"


class EvaluationStop(RuntimeError):
    """A controlled final-evaluation failure recorded in the manifest."""


class ReadOnlyTestYOLODataset(YOLODataset):
    """Redirect Ultralytics label-cache writes outside the held-out dataset."""

    def __init__(self, *args: Any, external_cache_path: Path, **kwargs: Any) -> None:
        self.external_cache_path = external_cache_path
        super().__init__(*args, **kwargs)

    def _load_or_scan_cache(self, cache_path: Path, cache_hash: str):
        del cache_path
        self.external_cache_path.parent.mkdir(parents=True, exist_ok=True)
        return super()._load_or_scan_cache(self.external_cache_path, cache_hash)


def resolve_split(data_yaml: Path, payload: dict[str, Any], split: str) -> Path:
    declared = payload.get(split)
    if not isinstance(declared, str) or not declared:
        raise EvaluationStop(f"Canonical dataset YAML has no valid {split!r} entry: {data_yaml}")
    path = Path(declared).expanduser()
    if not path.is_absolute():
        dataset_root = Path(payload.get("path") or data_yaml.parent).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = (data_yaml.parent / dataset_root).resolve()
        path = dataset_root / path
    path = path.resolve()
    if not path.is_dir():
        raise EvaluationStop(f"Canonical {split} images path does not exist: {path}")
    if path.name != "images" or not (path.parent / "labels").is_dir():
        raise EvaluationStop(f"Canonical {split} path is not an images directory with sibling labels: {path}")
    if not path.is_relative_to(ROOT):
        raise EvaluationStop(f"Canonical {split} path does not resolve under current project root: {path}")
    return path


def load_dataset_definition() -> tuple[dict[str, Any], dict[str, Path], Any]:
    data_yaml = (ROOT / DATA_YAML).resolve()
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise EvaluationStop(f"Invalid canonical dataset YAML: {data_yaml}")
    names_raw = raw.get("names")
    if isinstance(names_raw, list):
        names = dict(enumerate(names_raw))
    elif isinstance(names_raw, dict):
        names = {int(key): str(value) for key, value in names_raw.items()}
    else:
        raise EvaluationStop("Canonical dataset YAML has no valid class names.")
    if names != EXPECTED_NAMES:
        raise EvaluationStop(f"Class mapping changed: expected {EXPECTED_NAMES}, got {names}.")
    splits = {split: resolve_split(data_yaml, raw, split) for split in ("train", "val", "test")}
    base = yaml.safe_load((ROOT / BASE_CONFIG).read_text(encoding="utf-8"))
    dataset_version = base.get("dataset", {}).get("version") if isinstance(base, dict) else None
    data = {
        "path": str(splits["test"].parent.parent),
        "train": str(splits["train"]),
        "val": str(splits["val"]),
        "test": str(splits["test"]),
        "names": names,
        "nc": len(names),
        "channels": 3,
    }
    return data, splits, dataset_version


def evaluation_args() -> Any:
    return get_cfg(
        DEFAULT_CFG,
        {
            "task": "detect",
            "mode": "val",
            "split": "test",
            "imgsz": IMGSZ,
            "batch": BATCH,
            "workers": WORKERS,
            "device": "0",
            "cache": False,
            "fraction": 1.0,
            "rect": True,
            "augment": False,
            "amp": False,
            "plots": True,
            "save_json": False,
            "save_txt": False,
            "conf": None,
            "iou": 0.7,
            "verbose": True,
            "compile": False,
        },
    )


def build_test_dataset(
    model: torch.nn.Module,
    data: dict[str, Any],
    args: Any,
    external_cache_path: Path,
) -> Dataset:
    stride = max(int(model.stride.max()), 32)
    return ReadOnlyTestYOLODataset(
        img_path=data["test"],
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=False,
        hyp=copy.copy(args),
        rect=True,
        cache=False,
        single_cls=False,
        stride=stride,
        pad=0.5,
        prefix="test: ",
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
        external_cache_path=external_cache_path,
    )


def annotation_count(dataset: Dataset) -> int:
    return sum(int(len(label.get("cls", ()))) for label in dataset.labels)


def f1_score(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 0.0 if denominator == 0 else 2.0 * precision * recall / denominator


def trainer_context(model: torch.nn.Module, device: torch.device, data: dict[str, Any], args: Any) -> Any:
    zero = torch.zeros((), device=device)
    return SimpleNamespace(
        device=device,
        data=data,
        amp=False,
        ema=SimpleNamespace(ema=model),
        model=model,
        args=args,
        loss_items={"box_loss": zero, "cls_loss": zero, "dfl_loss": zero},
        world_size=1,
        label_loss_items=lambda loss_items, prefix="train": {
            f"{prefix}/{key}": round(float(value), 5) for key, value in loss_items.items()
        },
        stopper=SimpleNamespace(possible_stop=False),
        epoch=0,
        epochs=1,
    )


def extract_detection_metrics(validator: DetectionValidator) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    mean = [float(value) for value in validator.metrics.mean_results()]
    overall = {
        "precision": mean[0],
        "recall": mean[1],
        "mAP50": mean[2],
        "mAP50-95": mean[3],
        "F1": f1_score(mean[0], mean[1]),
    }
    per_class: dict[str, dict[str, float]] = {
        name: {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0, "F1": 0.0}
        for name in EXPECTED_NAMES.values()
    }
    for metric_index, class_id_raw in enumerate(validator.metrics.ap_class_index):
        class_id = int(class_id_raw)
        precision, recall, map50, map5095 = (
            float(value) for value in validator.metrics.class_result(metric_index)
        )
        per_class[EXPECTED_NAMES[class_id]] = {
            "precision": precision,
            "recall": recall,
            "mAP50": map50,
            "mAP50-95": map5095,
            "F1": f1_score(precision, recall),
        }
    return overall, per_class


def evaluate_native_model(
    label: str,
    model: torch.nn.Module,
    device: torch.device,
    data: dict[str, Any],
    output_dir: Path,
    quantizer_items: list[Any] | None = None,
) -> tuple[dict[str, Any], int, int]:
    args = evaluation_args()
    model.to(device).float().eval()
    dataset = build_test_dataset(model, data, args, output_dir.parent / "test_labels.cache")
    loader = build_dataloader(
        dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
        device=device,
    )
    validator = DetectionValidator(loader, save_dir=output_dir, args=copy.copy(args))
    calls: dict[int, int] = {}
    hooks: list[Any] = []
    if quantizer_items is not None:
        calls, hooks = install_quantizer_counter(quantizer_items)
    try:
        returned = validator(trainer_context(model, device, data, args))
    finally:
        for hook in hooks:
            hook.remove()
        if hasattr(loader, "close"):
            loader.close()
    overall, per_class = extract_detection_metrics(validator)
    count_images = len(dataset)
    count_annotations = annotation_count(dataset)
    fake_calls = sum(calls.values())
    if quantizer_items is not None and fake_calls < 1:
        raise EvaluationStop("QAT quantizers were present but no fake-quant operation executed during TEST evaluation.")
    report = {
        "model": label,
        "overall": overall,
        "per_class": per_class,
        "test_images": count_images,
        "test_annotations": count_annotations,
        "evaluator_return": {str(key): float(value) for key, value in (returned or {}).items()},
        "evaluator_configuration": {
            "split": "test",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "cache": False,
            "fraction": 1.0,
            "rect": True,
            "augment": False,
            "conf": validator.args.conf,
            "conf_source": "Ultralytics detection validation default resolved from conf=None; not tuned on TEST",
            "iou": validator.args.iou,
            "plots": True,
        },
        "timing": {
            "label": TIMING_LABEL,
            "unit": "milliseconds per image",
            "values": {key: float(value) for key, value in validator.speed.items()},
        },
        "fake_quant_calls": fake_calls if quantizer_items is not None else None,
    }
    return report, count_images, count_annotations


def load_fp32(path: Path, device: torch.device) -> torch.nn.Module:
    model = YOLO(str(path)).model
    attach_ultralytics_training_args(model, ROOT)
    if {int(key): str(value) for key, value in model.names.items()} != EXPECTED_NAMES:
        raise EvaluationStop(f"FP32 checkpoint class mapping differs from {EXPECTED_NAMES}.")
    return model.to(device)


def load_qat(path: Path, fp32_source: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "compression_state", "nncf_config"}
    missing = sorted(required.difference(saved))
    if missing:
        raise EvaluationStop(f"QAT checkpoint lacks NNCF restore state: {missing}")
    fp32_model = YOLO(str(fp32_source)).model
    attach_ultralytics_training_args(fp32_model, ROOT)
    restore_ultralytics_trainer_trainability(fp32_model)
    controller, qat_model = create_compressed_model(
        fp32_model,
        NNCFConfig.from_dict(saved["nncf_config"]),
        compression_state=saved["compression_state"],
        dump_graphs=False,
    )
    qat_model.to(device)
    loaded_entries = load_state(qat_model, saved["model_state_dict"], is_resume=True)
    items = quantizers(qat_model)
    if len(items) != EXPECTED_QUANTIZERS:
        raise EvaluationStop(f"Expected {EXPECTED_QUANTIZERS} QAT quantizers, found {len(items)}.")
    if not quantizers_enabled(items):
        raise EvaluationStop("QAT fake quantization is not active before TEST evaluation.")
    return qat_model, {
        "controller": controller,
        "quantizers": items,
        "loaded_state_entries": int(loaded_entries),
        "status": True,
    }


def delta_row(metric: str, fp32: float, qat: float, scope: str = "overall", class_name: str = "all") -> dict[str, Any]:
    delta = qat - fp32
    relative = None if fp32 == 0 else delta / fp32 * 100.0
    return {
        "scope": scope,
        "class": class_name,
        "metric": metric,
        "FP32": fp32,
        "QAT": qat,
        "absolute_delta": delta,
        "relative_delta_percent": relative,
    }


def build_comparison(fp32: dict[str, Any], qat: dict[str, Any]) -> dict[str, Any]:
    metrics = ("precision", "recall", "mAP50", "mAP50-95", "F1")
    overall = [delta_row(metric, fp32["overall"][metric], qat["overall"][metric]) for metric in metrics]
    per_class = [
        delta_row(metric, fp32["per_class"][name][metric], qat["per_class"][name][metric], "per_class", name)
        for name in EXPECTED_NAMES.values()
        for metric in metrics
    ]
    return {"overall": overall, "per_class": per_class, "statement": TEST_ONLY_STATEMENT}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    output_dir = ROOT / OUTPUT_DIR
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "started"}
    try:
        require_project_root(ROOT)
        fp32_path = (ROOT / FP32_CHECKPOINT).resolve()
        qat_path = (ROOT / QAT_CHECKPOINT).resolve()
        if not fp32_path.is_file() or not qat_path.is_file():
            raise EvaluationStop(f"Fixed final checkpoint missing: FP32={fp32_path}, QAT={qat_path}")
        device, gpu_name = require_cuda()
        if torch.cuda.device_count() != 1:
            raise EvaluationStop(f"Exactly one visible GPU is required, found {torch.cuda.device_count()}.")
        data, splits, dataset_version = load_dataset_definition()
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest.update(
            {
                "dataset version": dataset_version,
                "class mapping": EXPECTED_NAMES,
                "resolved dataset splits": {key: str(value) for key, value in splits.items()},
                "FP32 checkpoint path": str(fp32_path),
                "FP32 SHA256": sha256_file(fp32_path),
                "QAT checkpoint path": str(qat_path),
                "QAT SHA256": sha256_file(qat_path),
                "versions": {
                    "Python": platform.python_version(),
                    "torch": torch.__version__,
                    "torch CUDA": torch.version.cuda,
                    "torchvision": torchvision.__version__,
                    "ultralytics": ultralytics.__version__,
                    "nncf": nncf.__version__,
                },
                "GPU": {"logical_device": "cuda:0", "name": gpu_name},
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "imgsz": IMGSZ,
                "batch": BATCH,
                "workers": WORKERS,
                "exact evaluator configuration": {
                    "framework": "Ultralytics DetectionValidator native PyTorch path",
                    "split": "test",
                    "imgsz": IMGSZ,
                    "batch": BATCH,
                    "workers": WORKERS,
                    "device": "logical cuda:0",
                    "cache": False,
                    "augmentation": False,
                    "confidence": "Ultralytics detect validation default from conf=None (0.001), identical and untuned",
                    "iou": 0.7,
                    "rect": True,
                },
                "statement": TEST_ONLY_STATEMENT,
                "timing statement": TIMING_LABEL,
            }
        )
        write_json(manifest_path, manifest)
        print(f"Held-out TEST: {splits['test']}")
        print(f"FP32 checkpoint: {fp32_path}")
        print(f"QAT checkpoint: {qat_path}")

        fp32_model = load_fp32(fp32_path, device)
        fp32_report, test_images, test_annotations = evaluate_native_model(
            "FP32 PyTorch", fp32_model, device, data, output_dir / "fp32_plots"
        )
        write_json(output_dir / "fp32_test_metrics.json", fp32_report)
        del fp32_model
        torch.cuda.empty_cache()

        qat_model, reload_info = load_qat(qat_path, fp32_path, device)
        qat_report, qat_images, qat_annotations = evaluate_native_model(
            "QAT NNCF fake-quantized PyTorch",
            qat_model,
            device,
            data,
            output_dir / "qat_plots",
            reload_info["quantizers"],
        )
        write_json(output_dir / "qat_test_metrics.json", qat_report)
        if (test_images, test_annotations) != (qat_images, qat_annotations):
            raise EvaluationStop("FP32 and QAT evaluators did not observe identical TEST image/annotation counts.")

        comparison = build_comparison(fp32_report, qat_report)
        write_json(output_dir / "comparison.json", comparison)
        with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ("scope", "class", "metric", "FP32", "QAT", "absolute_delta", "relative_delta_percent")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(comparison["overall"] + comparison["per_class"])

        manifest.update(
            {
                "test image count": test_images,
                "test annotation count": test_annotations,
                "expected sanity reference": {"test images": 592, "test annotations": 734},
                "sanity reference matched": test_images == 592 and test_annotations == 734,
                "quantizer count for QAT": len(reload_info["quantizers"]),
                "NNCF reload status": reload_info["status"],
                "NNCF loaded state entries": reload_info["loaded_state_entries"],
                "QAT fake quantization active": True,
                "outputs": {
                    "FP32 metrics": str(output_dir / "fp32_test_metrics.json"),
                    "QAT metrics": str(output_dir / "qat_test_metrics.json"),
                    "comparison JSON": str(output_dir / "comparison.json"),
                    "comparison CSV": str(output_dir / "comparison.csv"),
                    "FP32 plots": str(output_dir / "fp32_plots"),
                    "QAT plots": str(output_dir / "qat_plots"),
                },
                "status": "passed",
            }
        )
        if not manifest["sanity reference matched"]:
            manifest["sanity note"] = "Actual counts differ from the historical reference; results use actual canonical TEST data."
        print(f"Final held-out TEST evaluation passed: {test_images} images, {test_annotations} annotations.")
    except (EvaluationStop, RuntimeError) as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        print(f"Final TEST evaluation failed: {manifest['error']}", file=sys.stderr)
        return_code = 1
    except Exception as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        print(f"Final TEST evaluation failed: {manifest['error']}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        write_json(manifest_path, manifest)
        print(f"Manifest: {manifest_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
