"""Prove NNCF PyTorch QAT compatibility for the YOLO11n FP32 checkpoint.

This is deliberately a *probe*, not a training job.  It uses only a handful
of images from the training split, performs range initialization on at most
one tiny batch, and executes exactly one optimizer step.  It never creates a
QAT checkpoint and never constructs a validation or test dataloader.

NNCF has two APIs that can look similar at a glance:

* ``nncf.quantize`` is the NNCF 2.13.0 post-training quantization API.
* ``nncf.torch.create_compressed_model`` inserts PyTorch fake quantizers and
  returns a model ready for compression-aware fine-tuning.  This probe uses
  only the latter.

Run this on the CUDA SSH host from the repository root, not in Codex's
sandbox.  CUDA_VISIBLE_DEVICES must be exported by the shell or loaded from
the repository .env before Python imports torch.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Do not set CUDA_VISIBLE_DEVICES here.  Shell exports stay authoritative and
# the project .env is only a fallback, matching the FP32 training script.
ROOT = Path.cwd().resolve()
load_dotenv(ROOT / ".env", override=False)

import nncf
import torch
import ultralytics
import yaml
from nncf import NNCFConfig
from nncf.config.extractors import extract_range_init_params
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.quantization.init_range import PTRangeInitParams
from nncf.torch.quantization.layers import BaseQuantizer
from torch.utils.data import DataLoader, Subset
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.data import build_yolo_dataset


CHECKPOINT_RELATIVE = Path("artifacts/checkpoints/fp32/best.pt")
REPORT_RELATIVE = Path("experiments/debug/qat_compatibility_probe.json")


class ProbeStop(RuntimeError):
    """A controlled stop that is recorded in the required JSON report."""


class YoloTrainInitializingDataLoader(PTInitializingDataLoader):
    """Teach NNCF how an Ultralytics detection training batch reaches model.forward."""

    def get_inputs(self, dataloader_output: dict[str, Any]):
        # DetectionTrainer.preprocess_batch applies this same normalization.
        # NNCF moves the returned tensor to the compressed model's device.
        return (dataloader_output["img"].float() / 255.0,), {}

    def get_target(self, dataloader_output: dict[str, Any]):
        return dataloader_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-images",
        type=Path,
        default=None,
        help=(
            "Training images directory. Defaults to the `train` value in "
            "data/dataset/data.yaml. Use this only if that YAML still has a "
            "stale absolute path after moving the project."
        ),
    )
    parser.add_argument("--batch", type=int, default=2, choices=(1, 2, 4), help="Tiny training batch size.")
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=4,
        help="Maximum train images exposed to this probe (must be >= batch).",
    )
    parser.add_argument(
        "--init-samples",
        type=int,
        default=2,
        help="Maximum training images used only to initialize quantizer ranges.",
    )
    return parser.parse_args()


def require_project_root(root: Path) -> None:
    if not (root / "pyproject.toml").is_file():
        raise ProbeStop("Run this script from the project root (the directory containing pyproject.toml).")


def require_cuda() -> tuple[torch.device, str]:
    # device=0 means the first *visible* CUDA device.  Visibility is supplied
    # externally; this script never changes CUDA_VISIBLE_DEVICES.
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ProbeStop("CUDA_VISIBLE_DEVICES is unset. Set it in the SSH shell or .env before running the probe.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ProbeStop(
            "CUDA is unavailable to this Python process. This is a host/environment issue; "
            "do not infer failure from a Codex sandbox run."
        )
    device = torch.device("cuda:0")
    return device, torch.cuda.get_device_name(0)


def resolve_train_images(root: Path, supplied_path: Path | None) -> Path:
    """Resolve only the train split and refuse stale paths rather than rewriting YAML."""
    if supplied_path is not None:
        train_images = supplied_path.expanduser()
        if not train_images.is_absolute():
            train_images = root / train_images
    else:
        data_yaml = root / "data/dataset/data.yaml"
        if not data_yaml.is_file():
            raise ProbeStop(f"Training dataset YAML not found: {data_yaml}")
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        declared_train = data.get("train") if isinstance(data, dict) else None
        if not declared_train:
            raise ProbeStop(f"No `train` entry found in {data_yaml}")
        train_images = Path(declared_train).expanduser()
        if not train_images.is_absolute():
            train_images = (data_yaml.parent / train_images).resolve()

    train_images = train_images.resolve()
    if not train_images.is_dir():
        raise ProbeStop(
            f"Training images directory does not exist: {train_images}. "
            "The dataset YAML/path was not edited by this probe. Pass --train-images with the current TRAIN path "
            "or repair data/dataset/data.yaml separately."
        )
    return train_images


def load_names(root: Path) -> dict[int, str]:
    """Read class names without asking Ultralytics to inspect val/test paths."""
    data_yaml = root / "data/dataset/data.yaml"
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ProbeStop(f"`names` is missing or invalid in {data_yaml}")
    return {int(key): str(value) for key, value in names.items()}


def attach_ultralytics_training_args(model: torch.nn.Module, root: Path) -> None:
    """Restore the trainer-provided namespace needed by DetectionModel.loss().

    A serialized Ultralytics checkpoint can carry ``model.args`` as a plain
    dict.  Installed DetectionTrainer.set_model_attributes() instead assigns
    its namespace (`self.model.args = self.args`) before loss construction.
    The probe reproduces only that loss prerequisite, using the three loss
    gains stored by the FP32 baseline; it does not instantiate a trainer.
    """
    baseline_args_path = root / "artifacts/checkpoints/fp32/args.yaml"
    baseline_args = yaml.safe_load(baseline_args_path.read_text(encoding="utf-8"))
    if not isinstance(baseline_args, dict):
        raise ProbeStop(f"Invalid baseline training arguments: {baseline_args_path}")
    required_gains = ("box", "cls", "dfl")
    if any(key not in baseline_args for key in required_gains):
        raise ProbeStop(f"Baseline training arguments lack one of {required_gains}: {baseline_args_path}")
    model.args = get_cfg(
        DEFAULT_CFG,
        {"task": "detect", **{key: float(baseline_args[key]) for key in required_gains}},
    )
    # Ensure the criterion is constructed after model.args is a namespace.
    model.criterion = None


def restore_ultralytics_trainer_trainability(model: torch.nn.Module) -> dict[str, Any]:
    """Apply the relevant BaseTrainer._setup_train() requires-grad policy.

    ``load_checkpoint()`` returns an inference-oriented model. Installed
    BaseTrainer._setup_train() restores gradients for floating-point parameters
    after loading, while keeping ``.dfl`` frozen.  QAT fine-tuning must follow
    that same policy before NNCF inserts fake quantizers.
    """
    always_frozen = ".dfl"
    restored = 0
    intentionally_frozen = 0
    for name, parameter in model.named_parameters():
        if always_frozen in name:
            parameter.requires_grad = False
            intentionally_frozen += parameter.numel()
        elif not parameter.requires_grad and parameter.dtype.is_floating_point:
            parameter.requires_grad = True
            restored += parameter.numel()
    return {
        "policy": "Ultralytics BaseTrainer._setup_train(): restore floating-point gradients; freeze .dfl only",
        "parameters_restored": restored,
        "parameters_intentionally_frozen": intentionally_frozen,
        "trainable_parameters_before_nncf": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def build_tiny_train_loader(
    model: torch.nn.Module, train_images: Path, names: dict[int, str], batch_size: int, max_train_images: int
) -> DataLoader:
    # This directly uses Ultralytics' installed dataset builder.  DetectionTrainer.get_dataloader()
    # would be unnecessary here and its BaseTrainer constructor calls get_dataset(), which resolves val too.
    cfg = get_cfg(
        DEFAULT_CFG,
        {
            "task": "detect",
            "imgsz": 640,
            "workers": 0,
            "cache": False,
            "rect": False,
            "fraction": 1.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
        },
    )
    stride = max(int(model.stride.max()), 32)
    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=str(train_images),
        batch=batch_size,
        data={"names": names, "nc": len(names), "channels": 3},
        mode="train",
        rect=False,
        stride=stride,
    )
    if len(dataset) < 1:
        raise ProbeStop(f"No train images were found in {train_images}")
    limited_count = min(len(dataset), max_train_images)
    if limited_count < batch_size:
        raise ProbeStop(f"Only {limited_count} train image(s) available, fewer than requested --batch={batch_size}.")
    return DataLoader(
        Subset(dataset, range(limited_count)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )


def preprocess_yolo_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """The non-multiscale portion of DetectionTrainer.preprocess_batch()."""
    return {
        key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
        for key, value in batch.items()
    } | {"img": batch["img"].to(device, non_blocking=True).float() / 255.0}


def api_report() -> dict[str, Any]:
    """Runtime inspection prevents substituting a differently-versioned online API."""
    ptq_quantize = getattr(nncf, "quantize", None)
    training_doc = inspect.getdoc(create_compressed_model) or ""
    ptq_doc = inspect.getdoc(ptq_quantize) if ptq_quantize is not None else ""
    report = {
        "training_time_api": "nncf.torch.create_compressed_model",
        "training_time_purpose_from_installed_docstring": training_doc.splitlines()[0] if training_doc else None,
        "initialization_api": "nncf.torch.register_default_init_args",
        "runtime_range_initialization_api": "QuantizationController.init_range(PTRangeInitParams)",
        "ptq_api_not_used": "nncf.quantize",
        "ptq_purpose_from_installed_docstring": ptq_doc.splitlines()[0] if ptq_doc else "unavailable",
        "available": {
            "nncf.torch.create_compressed_model": hasattr(nncf.torch, "create_compressed_model"),
            "nncf.torch.register_default_init_args": hasattr(nncf.torch, "register_default_init_args"),
            "nncf.quantize": ptq_quantize is not None,
        },
        "signatures": {
            "nncf.torch.create_compressed_model": str(inspect.signature(create_compressed_model)),
            "nncf.torch.register_default_init_args": str(inspect.signature(register_default_init_args)),
            "nncf.quantize": str(inspect.signature(ptq_quantize)) if ptq_quantize is not None else None,
        },
        "source_files": {
            "create_compressed_model": inspect.getsourcefile(create_compressed_model),
            "register_default_init_args": inspect.getsourcefile(register_default_init_args),
            "quantize": inspect.getsourcefile(ptq_quantize) if ptq_quantize is not None else None,
        },
    }
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serialize_loss_items(loss_items: Any) -> dict[str, float] | list[float] | str | None:
    """Handle both installed Ultralytics loss-item conventions reproducibly."""
    if loss_items is None:
        return None
    if isinstance(loss_items, Mapping):
        return {
            str(key): float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in loss_items.items()
        }
    if isinstance(loss_items, torch.Tensor):
        return [float(value) for value in loss_items.detach().cpu().reshape(-1).tolist()]
    return str(loss_items)


def main() -> int:
    args = parse_args()
    require_project_root(ROOT)
    checkpoint = (ROOT / CHECKPOINT_RELATIVE).resolve()
    report_path = ROOT / REPORT_RELATIVE
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "nncf": nncf.__version__,
        },
        "checkpoint": str(checkpoint),
        "NNCF API used": api_report(),
        "quantizer_count": 0,
        "quantizer_types": {},
        "forward_status": False,
        "backward_status": False,
        "optimizer_step_status": False,
        "gpu": None,
        "notes": [
            "QAT compatibility probe only: train split only, no validation/test loader, no PTQ, no checkpoint save.",
            "At most one optimizer step is executed.",
        ],
    }

    try:
        device, gpu_name = require_cuda()
        report["gpu"] = {
            "logical_device": "cuda:0",
            "name": gpu_name,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda": torch.version.cuda,
        }
        if not checkpoint.is_file():
            raise ProbeStop(f"FP32 checkpoint not found: {checkpoint}")
        if args.init_samples < 1 or args.init_samples > args.max_train_images:
            raise ProbeStop("--init-samples must be >= 1 and <= --max-train-images.")

        train_images = resolve_train_images(ROOT, args.train_images)
        names = load_names(ROOT)
        print(f"Python: {report['versions']['python']}")
        print(f"torch: {torch.__version__}")
        print(f"ultralytics: {ultralytics.__version__}")
        print(f"nncf: {nncf.__version__}")
        print(f"GPU (device=0): {gpu_name}")
        print(f"Checkpoint: {checkpoint}")
        print("NNCF QAT API: nncf.torch.create_compressed_model")
        print("NNCF PTQ API intentionally not used: nncf.quantize")
        print(f"Train images only: {train_images}")

        # YOLO(...).model is the checkpoint-loaded DetectionModel used by the normal Ultralytics trainer.
        # The checkpoint file is read only; no model/checkpoint is overwritten by this script.
        yolo = YOLO(str(checkpoint))
        fp32_model = yolo.model.to(device)
        attach_ultralytics_training_args(fp32_model, ROOT)
        report["ultralytics_trainer_trainability"] = restore_ultralytics_trainer_trainability(fp32_model)
        train_loader = build_tiny_train_loader(fp32_model, train_images, names, args.batch, args.max_train_images)
        init_loader = YoloTrainInitializingDataLoader(train_loader)

        # NNCF 2.13.0 first builds a temporary clean model to collect build-time range
        # statistics. For this YOLO checkpoint that temporary collector contains an
        # unexecuted graph point and raises on a missing statistic. Create the genuine
        # fake-quantized NNCFNetwork first, then use the controller's documented
        # init_range() path. It performs the same range initialization against the
        # inserted quantizers and still consumes only the train-only tiny loader.
        # nncf.quantize is never called.
        nncf_config = NNCFConfig.from_dict(
            {
                # Match the fixed, drop_last probe loader batch exactly so YOLO's
                # shape-cached Detect head follows one stable graph branch.
                "input_info": {"sample_size": [args.batch, 3, 640, 640]},
                "compression": {"algorithm": "quantization"},
            }
        )
        compression_ctrl, qat_model = create_compressed_model(fp32_model, nncf_config, dump_graphs=False)
        qat_model.to(device)

        runtime_range_config = NNCFConfig.from_dict(
            {
                "input_info": {"sample_size": [args.batch, 3, 640, 640]},
                "compression": {
                    "algorithm": "quantization",
                    "initializer": {"range": {"num_init_samples": args.init_samples}},
                },
            }
        )
        register_default_init_args(runtime_range_config, init_loader, device=str(device))
        runtime_range_params_dict = extract_range_init_params(runtime_range_config, "quantization")
        if runtime_range_params_dict is None:
            raise ProbeStop("NNCF did not produce runtime range-initialization parameters.")
        # NNCF's installed extractor returns constructor kwargs rather than the
        # PTRangeInitParams object required by QuantizationController.init_range.
        runtime_range_params = PTRangeInitParams(**runtime_range_params_dict)
        compression_ctrl.init_range(runtime_range_params)
        report["range_initialization"] = {
            "method": "QuantizationController.init_range",
            "train_samples_requested": args.init_samples,
            "train_loader_batch_size": args.batch,
        }
        report["notes"].append(
            "DetectionModel.loss() uses the installed trainer's model.args namespace convention with FP32 baseline box/cls/dfl gains."
        )

        quantizers = [module for module in qat_model.modules() if isinstance(module, BaseQuantizer)]
        report["quantizer_count"] = len(quantizers)
        report["quantizer_types"] = dict(sorted(Counter(type(module).__name__ for module in quantizers).items()))
        report["controller_quantizers"] = {
            "all": len(compression_ctrl.all_quantizations),
            "weight": len(compression_ctrl.weight_quantizers),
            "non_weight": len(compression_ctrl.non_weight_quantizers),
        }
        if not quantizers:
            raise ProbeStop("NNCF returned no BaseQuantizer modules; refusing to claim QAT compatibility.")

        quantizer_calls = {id(module): 0 for module in quantizers}
        hooks = [
            module.register_forward_hook(lambda _m, _inputs, _output, key=id(module): quantizer_calls.__setitem__(key, quantizer_calls[key] + 1))
            for module in quantizers
        ]
        try:
            raw_batch = next(iter(train_loader))
            batch = preprocess_yolo_batch(raw_batch, device)
            if tuple(batch["img"].shape[-2:]) != (640, 640):
                raise ProbeStop(f"Expected imgsz=640 batch, got {tuple(batch['img'].shape[-2:])}")

            qat_model.train()
            optimizer = torch.optim.AdamW((p for p in qat_model.parameters() if p.requires_grad), lr=1e-6)
            optimizer.zero_grad(set_to_none=True)
            predictions = qat_model(batch["img"])
            report["forward_status"] = True
            raw_loss, loss_items = qat_model.loss(batch, preds=predictions)
            # Ultralytics 8.4.130 in this environment returns a three-component
            # (box/cls/dfl) loss vector.  The training loop backpropagates its
            # scalar total; preserve components in the report and sum explicitly.
            if not isinstance(raw_loss, torch.Tensor):
                raise ProbeStop(f"YOLO detection loss must be a Tensor, got {type(raw_loss).__name__}.")
            if not torch.isfinite(raw_loss).all().item():
                raise ProbeStop(f"YOLO detection loss contains non-finite values: {raw_loss.detach().cpu().tolist()}")
            loss = raw_loss.sum()
            loss.backward()
            report["backward_status"] = True
            optimizer.step()  # The sole and final optimizer step in this probe.
            report["optimizer_step_status"] = True
            report["loss"] = float(loss.detach().cpu())
            report["loss_components"] = [float(value) for value in raw_loss.detach().cpu().reshape(-1).tolist()]
            report["loss_items"] = serialize_loss_items(loss_items)
        finally:
            for hook in hooks:
                hook.remove()

        report["fake_quantizer_forward_calls"] = sum(quantizer_calls.values())
        report["trainable_parameters"] = sum(parameter.numel() for parameter in qat_model.parameters() if parameter.requires_grad)
        if report["fake_quantizer_forward_calls"] < 1:
            raise ProbeStop("BaseQuantizer modules were present but none executed in the imgsz=640 forward pass.")
        report["status"] = "passed"
        report["notes"].append(
            "The NNCF-wrapped DetectionModel accepted Ultralytics DetectionModel.loss(batch, preds), and one backward/step succeeded."
        )
    except ProbeStop as error:
        report["status"] = "stopped"
        report["notes"].append(str(error))
        write_report(report_path, report)
        print(f"QAT compatibility probe stopped: {error}", file=sys.stderr)
        print(f"Report: {report_path}", file=sys.stderr)
        return 2
    except Exception as error:
        report["status"] = "failed"
        report["notes"].append(f"{type(error).__name__}: {error}")
        write_report(report_path, report)
        print(f"QAT compatibility probe failed: {type(error).__name__}: {error}", file=sys.stderr)
        print(f"Report: {report_path}", file=sys.stderr)
        raise
    else:
        write_report(report_path, report)
        print("QAT compatibility probe passed (no QAT checkpoint was saved).")
        print(f"Report: {report_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
