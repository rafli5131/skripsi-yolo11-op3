"""One-epoch NNCF QAT smoke test through Ultralytics DetectionTrainer.

This is intentionally *not* the Phase 5 QAT experiment.  It proves the
training lifecycle only:

    FP32 best.pt -> NNCF fake quantizers -> DetectionTrainer -> train/val
    -> NNCF-aware checkpoint -> independent reload/validation.

Only train and validation paths are declared in the generated smoke dataset
YAML.  The test split is never read.  ``nncf.quantize`` is deliberately not
imported or called: it is NNCF's PTQ API, not the proven PyTorch QAT path.

Run from the repository root on the CUDA SSH host.  CUDA_VISIBLE_DEVICES is
loaded from the shell or .env before torch is imported and is never changed by
this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
# This only disables the optional W&B integration.  In particular, it does
# not alter CUDA_VISIBLE_DEVICES, whose externally supplied value is recorded.
os.environ["WANDB_MODE"] = "disabled"

import nncf
import torch
import ultralytics
import yaml
from nncf import NNCFConfig
from nncf.config.extractors import extract_range_init_params
from nncf.torch import create_compressed_model, load_state, register_default_init_args
from nncf.torch.quantization.init_range import PTRangeInitParams
from nncf.torch.quantization.layers import BaseQuantizer
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

# Phase 5A's train-only loader and loss/trainability handling are deliberately
# reused instead of reimplementing a second, potentially incompatible path.
from probe_qat_compatibility import (  # noqa: E402
    ProbeStop,
    YoloTrainInitializingDataLoader,
    attach_ultralytics_training_args,
    build_tiny_train_loader,
    load_names,
    require_cuda,
    require_project_root,
    restore_ultralytics_trainer_trainability,
)


CHECKPOINT_RELATIVE = Path("artifacts/checkpoints/fp32/best.pt")
SMOKE_RELATIVE = Path("experiments/debug/qat_smoke")
DEFAULT_TRAIN_IMAGES = Path("data/raw/advance_vision_human/train/images")
DEFAULT_VAL_IMAGES = Path("data/raw/advance_vision_human/valid/images")


class SmokeStop(RuntimeError):
    """A controlled stop that is written to the smoke manifest."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", type=Path, default=DEFAULT_TRAIN_IMAGES)
    parser.add_argument("--val-images", type=Path, default=DEFAULT_VAL_IMAGES)
    parser.add_argument("--range-init-samples", type=int, default=8)
    parser.add_argument("--range-init-batch", type=int, default=2)
    parser.add_argument(
        "--reload-only",
        action="store_true",
        help="Verify an existing weights/qat_last.pt without starting another training epoch.",
    )
    return parser.parse_args()


def resolve_images(path: Path, split: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    resolved = resolved.resolve()
    if not resolved.is_dir():
        raise SmokeStop(f"{split} images directory does not exist: {resolved}")
    if resolved.name != "images":
        raise SmokeStop(f"{split} path must be an Ultralytics images directory, got: {resolved}")
    labels = resolved.parent / "labels"
    if not labels.is_dir():
        raise SmokeStop(f"{split} labels directory does not exist beside images: {labels}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def write_smoke_data_yaml(path: Path, train_images: Path, val_images: Path, names: dict[int, str]) -> None:
    """Create a run-local data descriptor with no test key by design."""
    payload = {
        "train": str(train_images),
        "val": str(val_images),
        "nc": len(names),
        "names": names,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def make_nncf_structure_config(batch_size: int) -> NNCFConfig:
    """Return the fake-quant structure config proven in Phase 5A.

    Range initialization intentionally happens after fake-quant insertion via
    QuantizationController.init_range(), because NNCF 2.13.0's build-time
    collector hit an unexecuted YOLO graph point in Phase 5A.
    """
    return NNCFConfig.from_dict(
        {
            "input_info": {"sample_size": [batch_size, 3, 640, 640]},
            "compression": {"algorithm": "quantization"},
        }
    )


def initialize_ranges(
    controller: Any,
    model: torch.nn.Module,
    train_images: Path,
    names: dict[int, str],
    device: torch.device,
    init_batch: int,
    init_samples: int,
) -> dict[str, Any]:
    """Use the Phase 5A runtime range-init API on an explicit TRAIN subset."""
    if init_samples < init_batch:
        raise SmokeStop("--range-init-samples must be at least --range-init-batch.")
    tiny_loader = build_tiny_train_loader(model, train_images, names, init_batch, init_samples)
    init_loader = YoloTrainInitializingDataLoader(tiny_loader)
    runtime_config = NNCFConfig.from_dict(
        {
            "input_info": {"sample_size": [init_batch, 3, 640, 640]},
            "compression": {
                "algorithm": "quantization",
                "initializer": {"range": {"num_init_samples": init_samples}},
            },
        }
    )
    register_default_init_args(runtime_config, init_loader, device=str(device))
    init_params_dict = extract_range_init_params(runtime_config, "quantization")
    if init_params_dict is None:
        raise SmokeStop("NNCF did not produce PTRangeInitParams for runtime range initialization.")
    controller.init_range(PTRangeInitParams(**init_params_dict))
    return {
        "method": "QuantizationController.init_range(PTRangeInitParams)",
        "initializing_loader": "YoloTrainInitializingDataLoader over explicit TRAIN subset",
        "sample_count_requested": init_samples,
        "batch_size": init_batch,
    }


def quantizers(model: torch.nn.Module) -> list[BaseQuantizer]:
    return [module for module in model.modules() if isinstance(module, BaseQuantizer)]


def quantizers_enabled(items: list[BaseQuantizer]) -> bool:
    # BaseQuantizer exposes is_enabled_quantization() in NNCF 2.13.0.  Keep
    # the check explicit so a present-but-disabled structure cannot pass.
    return bool(items) and all(item.is_enabled_quantization() for item in items)


def install_quantizer_counter(items: list[BaseQuantizer]) -> tuple[dict[int, int], list[Any]]:
    calls = {id(item): 0 for item in items}
    hooks = [
        item.register_forward_hook(
            lambda _module, _inputs, _output, item_id=id(item): calls.__setitem__(item_id, calls[item_id] + 1)
        )
        for item in items
    ]
    return calls, hooks


def metric_value(metrics: dict[str, Any], suffix: str) -> float | None:
    for key, value in metrics.items():
        normalized = str(key).lower().replace(" ", "")
        if suffix in normalized:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def build_reload_validation_context(
    reloaded_model: torch.nn.Module,
    device: torch.device,
    train_images: Path,
    val_images: Path,
    names: dict[int, str],
    save_dir: Path,
) -> tuple[DetectionValidator, SimpleNamespace]:
    """Build the installed val loader without AutoBackend or a test split.

    DetectionValidator(model=...) takes its standalone AutoBackend path.  That
    path is correct for ordinary exports, but it is not an NNCF resume path and
    attempted to fuse/wrap this native fake-quantized model.  Supplying the
    same small trainer-shaped context used by BaseTrainer.validate() keeps the
    reloaded NNCFNetwork native and still uses Ultralytics' real validator.
    """
    validation_args = get_cfg(
        DEFAULT_CFG,
        {
            "task": "detect",
            "mode": "val",
            "imgsz": 640,
            "batch": 16,  # DetectionTrainer uses 2 * train batch for validation.
            "workers": 2,
            "fraction": 0.10,
            "cache": False,
            "plots": False,
            "device": "0",
            "amp": False,
            "split": "val",
            "rect": True,
            "compile": False,
        },
    )
    data = {
        "train": str(train_images),
        "val": str(val_images),
        "names": names,
        "nc": len(names),
        "channels": 3,
    }
    stride = max(int(reloaded_model.stride.max()), 32)
    dataset = build_yolo_dataset(
        validation_args,
        img_path=str(val_images),
        batch=validation_args.batch,
        data=data,
        mode="val",
        rect=True,
        stride=stride,
    )
    dataloader = build_dataloader(
        dataset,
        batch=validation_args.batch,
        workers=validation_args.workers * 2,
        shuffle=False,
        rank=-1,
        device=device,
    )
    validator = DetectionValidator(dataloader, save_dir=save_dir, args=copy.copy(validation_args))
    zeros = torch.zeros((), device=device)
    context = SimpleNamespace(
        device=device,
        data=data,
        amp=False,
        ema=SimpleNamespace(ema=reloaded_model),
        model=reloaded_model,
        args=validation_args,
        loss_items={"box_loss": zeros, "cls_loss": zeros, "dfl_loss": zeros},
        world_size=1,
        label_loss_items=lambda loss_items, prefix="train": {
            f"{prefix}/{key}": round(float(value), 5) for key, value in loss_items.items()
        },
        stopper=SimpleNamespace(possible_stop=False),
        epoch=0,
        epochs=1,
    )
    return validator, context


class QATSmokeDetectionTrainer(DetectionTrainer):
    """DetectionTrainer with one supported, exactly-once NNCF model preparation.

    BaseTrainer.setup_model() calls ``get_model()`` before it builds train/val
    loaders.  This override therefore loads the requested trained FP32
    DetectionModel itself, applies NNCF exactly once, and uses a separate
    tiny TRAIN-only loader for range initialization.  The rest of the epoch,
    optimizer, validation, and dataloaders are the installed trainer lifecycle.
    """

    def __init__(
        self,
        *args: Any,
        source_checkpoint: Path,
        train_images: Path,
        names: dict[int, str],
        range_init_batch: int,
        range_init_samples: int,
        **kwargs: Any,
    ) -> None:
        self.source_checkpoint = source_checkpoint
        self.qat_train_images = train_images
        self.qat_names = names
        self.range_init_batch = range_init_batch
        self.range_init_samples = range_init_samples
        self.compression_ctrl: Any | None = None
        self.nncf_structure_config: NNCFConfig | None = None
        self.range_init_report: dict[str, Any] | None = None
        self.qat_model_prepared = False
        self.quantizer_count_before_training = 0
        self.fake_quant_enabled_before_training = False
        self.fake_quant_calls: dict[int, int] | None = None
        self._fake_quant_hooks: list[Any] = []
        self.parameter_fingerprint: dict[str, str] | None = None
        self.qat_checkpoint_path: Path | None = None
        self.qat_best_checkpoint_path: Path | None = None
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):  # type: ignore[override]
        del cfg, weights, verbose  # The source checkpoint is the only allowed model source.
        if self.qat_model_prepared:
            raise SmokeStop("Refusing a second create_compressed_model() call on the same smoke trainer.")

        fp32_model = YOLO(str(self.source_checkpoint)).model
        if int(getattr(fp32_model, "nc", -1)) != int(self.data["nc"]):
            raise SmokeStop(
                f"Source checkpoint nc={getattr(fp32_model, 'nc', None)} does not match smoke dataset nc={self.data['nc']}."
            )
        # The loaded checkpoint carries an inference-style args dict.  Restore
        # loss args and the exact .dfl freeze policy before NNCF wraps it.
        attach_ultralytics_training_args(fp32_model, ROOT)
        restore_ultralytics_trainer_trainability(fp32_model)

        self.nncf_structure_config = make_nncf_structure_config(self.range_init_batch)
        controller, qat_model = create_compressed_model(fp32_model, self.nncf_structure_config, dump_graphs=False)
        qat_model.to(self.device)
        self.range_init_report = initialize_ranges(
            controller,
            qat_model,
            self.qat_train_images,
            self.qat_names,
            self.device,
            self.range_init_batch,
            self.range_init_samples,
        )
        items = quantizers(qat_model)
        self.quantizer_count_before_training = len(items)
        self.fake_quant_enabled_before_training = quantizers_enabled(items)
        if self.quantizer_count_before_training < 1 or not self.fake_quant_enabled_before_training:
            raise SmokeStop("NNCF fake quantizers are absent or disabled after model preparation.")
        print(
            "QAT model prepared before trainer epoch: "
            f"quantizers={self.quantizer_count_before_training}; "
            f"fake_quant_enabled={self.fake_quant_enabled_before_training}",
            flush=True,
        )

        # Count real calls across the trainer forward/validation lifecycle.
        self.fake_quant_calls, self._fake_quant_hooks = install_quantizer_counter(items)

        # Store a trainable tensor fingerprint before the epoch.  The reload
        # path compares it to the saved trained value, proving it did not fall
        # back to the FP32 initialization.
        for name, parameter in qat_model.named_parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                self.parameter_fingerprint = {"name": name, "before": tensor_sha256(parameter)}
                break
        if self.parameter_fingerprint is None:
            raise SmokeStop("No trainable floating-point QAT parameter was found.")

        self.compression_ctrl = controller
        self.qat_model_prepared = True
        return qat_model

    def save_model(self) -> bool:  # type: ignore[override]
        """Save a non-pickled, NNCF-reconstructible checkpoint after validation.

        Installed BaseTrainer.save_model() serializes an EMA model as a normal
        Ultralytics checkpoint.  NNCF's documented resume mechanism additionally
        requires controller compression state passed into create_compressed_model;
        therefore this smoke saves an explicit payload instead.
        """
        if self.compression_ctrl is None or self.nncf_structure_config is None:
            raise SmokeStop("Cannot save QAT state before NNCF preparation.")
        model = unwrap_model(self.model)
        state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        fingerprint = dict(self.parameter_fingerprint or {})
        if fingerprint.get("name") not in state_dict:
            raise SmokeStop("The QAT fingerprint parameter is missing from the NNCF state dictionary.")
        fingerprint["after"] = tensor_sha256(state_dict[fingerprint["name"]])
        payload = {
            "format": "nncf_qat_smoke_v1",
            "epoch": int(self.epoch),
            "source_fp32_checkpoint": str(self.source_checkpoint),
            "model_state_dict": state_dict,
            # NNCF 2.13.0 BaseController.get_compression_state() documents
            # this exact state for create_compressed_model(compression_state=...).
            "compression_state": copy.deepcopy(self.compression_ctrl.get_compression_state()),
            "nncf_config": copy.deepcopy(dict(self.nncf_structure_config)),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "train_args": vars(self.args).copy(),
            "metrics": dict(self.metrics or {}),
            "parameter_fingerprint": fingerprint,
        }
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.qat_checkpoint_path = self.wdir / "qat_last.pt"
        torch.save(payload, self.qat_checkpoint_path)
        if self.best_fitness == self.fitness:
            self.qat_best_checkpoint_path = self.wdir / "qat_best.pt"
            torch.save(payload, self.qat_best_checkpoint_path)
        return True

    def final_eval(self) -> None:  # type: ignore[override]
        """Avoid treating ordinary Ultralytics best.pt as an NNCF resume file.

        The epoch already executes BaseTrainer.validate().  Reload verification
        below validates the explicit NNCF checkpoint, so a second ordinary-pt
        final evaluation would be both wrong and redundant.
        """
        return None

    def remove_quantizer_hooks(self) -> int:
        for hook in self._fake_quant_hooks:
            hook.remove()
        self._fake_quant_hooks.clear()
        return sum((self.fake_quant_calls or {}).values())


def reload_and_validate(
    checkpoint_path: Path,
    source_checkpoint: Path,
    device: torch.device,
    trainer: QATSmokeDetectionTrainer | None,
    train_images: Path,
    val_images: Path,
    names: dict[int, str],
    save_dir: Path,
) -> dict[str, Any]:
    """Recreate structure, restore controller state/weights, then validate it."""
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "compression_state", "nncf_config", "parameter_fingerprint"}
    missing = sorted(required.difference(saved))
    if missing:
        raise SmokeStop(f"QAT checkpoint misses required NNCF state: {missing}")

    fp32_model = YOLO(str(source_checkpoint)).model
    attach_ultralytics_training_args(fp32_model, ROOT)
    restore_ultralytics_trainer_trainability(fp32_model)
    restored_config = NNCFConfig.from_dict(saved["nncf_config"])
    controller, reloaded_model = create_compressed_model(
        fp32_model,
        restored_config,
        compression_state=saved["compression_state"],
        dump_graphs=False,
    )
    reloaded_model.to(device)
    loaded_count = load_state(reloaded_model, saved["model_state_dict"], is_resume=True)
    items = quantizers(reloaded_model)
    enabled = quantizers_enabled(items)
    if not items or not enabled:
        raise SmokeStop("Reloaded model does not retain enabled fake quantizers.")

    fingerprint = saved["parameter_fingerprint"]
    reloaded_state = reloaded_model.state_dict()
    selected_name = fingerprint.get("name")
    if selected_name not in reloaded_state:
        raise SmokeStop("Reloaded model does not contain the saved trained-parameter fingerprint.")
    restored_hash = tensor_sha256(reloaded_state[selected_name])
    if restored_hash != fingerprint.get("after"):
        raise SmokeStop("Reloaded parameter fingerprint does not match the saved QAT state.")
    if fingerprint.get("before") == fingerprint.get("after"):
        raise SmokeStop("The fingerprinted QAT parameter did not change during the smoke epoch.")

    calls, hooks = install_quantizer_counter(items)
    try:
        reloaded_model.eval()
        with torch.no_grad():
            output = reloaded_model(torch.zeros((1, 3, 640, 640), device=device))
        if output is None:
            raise SmokeStop("Reloaded QAT model returned no output for imgsz=640 forward.")
        forward_calls = sum(calls.values())
        if forward_calls < 1:
            raise SmokeStop("Reloaded fake quantizers did not execute in the imgsz=640 forward.")

        # BaseTrainer closes its train/val workers at the end of train().
        # Always construct a fresh val-only loader here, and use the same
        # trainer-shaped native validation context.  Calling validator(model=...)
        # would enter AutoBackend and is not an NNCF resume path.
        validator, validation_context = build_reload_validation_context(
            reloaded_model, device, train_images, val_images, names, save_dir
        )
        validation_metrics = validator(validation_context)
    finally:
        for hook in hooks:
            hook.remove()

    return {
        "status": True,
        "checkpoint": str(checkpoint_path),
        "loaded_state_entries": int(loaded_count),
        "quantizer_count": len(items),
        "fake_quant_enabled": enabled,
        "forward_640_status": True,
        "fake_quant_calls_forward_640": forward_calls,
        "validation_inference_status": validation_metrics is not None,
        "trained_weights_restored": True,
        "trained_parameter_fingerprint": {
            "name": selected_name,
            "saved_hash": fingerprint.get("after"),
            "reloaded_hash": restored_hash,
            "changed_from_fp32_initialization": True,
        },
        "controller_type": type(controller).__name__,
        "validation_metrics": dict(validation_metrics or {}),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def reload_only(args: argparse.Namespace) -> int:
    """Finish reload verification for an already-completed one-epoch smoke."""
    manifest_path = ROOT / SMOKE_RELATIVE / "manifest.json"
    checkpoint_path = ROOT / SMOKE_RELATIVE / "weights" / "qat_last.pt"
    try:
        require_project_root(ROOT)
        if not manifest_path.is_file() or not checkpoint_path.is_file():
            raise SmokeStop("--reload-only requires an existing qat_smoke manifest and weights/qat_last.pt.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_checkpoint = (ROOT / CHECKPOINT_RELATIVE).resolve()
        device, gpu_name = require_cuda()
        train_images = resolve_images(args.train_images, "train")
        val_images = resolve_images(args.val_images, "validation")
        names = load_names(ROOT)
        print(f"Reload-only verification: {checkpoint_path}")
        reload_report = reload_and_validate(
            checkpoint_path,
            source_checkpoint,
            device,
            None,
            train_images,
            val_images,
            names,
            ROOT / SMOKE_RELATIVE,
        )
        manifest.update(
            {
                "reload": reload_report,
                "quantizer count after reload": reload_report["quantizer_count"],
                "reload status": reload_report["status"],
                "status": "passed",
                "reload_verified_at": datetime.now(timezone.utc).isoformat(),
                "GPU": {"logical_device": "cuda:0", "name": gpu_name},
            }
        )
        manifest.pop("error", None)
        write_manifest(manifest_path, manifest)
        print("QAT checkpoint reload and validation passed (no training epoch was run).")
        print(f"Manifest: {manifest_path}")
        return 0
    except (SmokeStop, ProbeStop) as error:
        message = f"{type(error).__name__}: {error}"
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        manifest.update({"reload status": False, "status": "failed", "error": message})
        write_manifest(manifest_path, manifest)
    finally:
        print(f"QAT reload-only verification failed: {message}", file=sys.stderr)
        print(f"Manifest: {manifest_path}")
    return 1


def main() -> int:
    args = parse_args()
    if args.reload_only:
        return reload_only(args)
    manifest_path = ROOT / SMOKE_RELATIVE / "manifest.json"
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "source_fp32_checkpoint": str((ROOT / CHECKPOINT_RELATIVE).resolve()),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "nncf": nncf.__version__,
        },
        "epochs": 1,
        "fraction": 0.10,
        "batch": 8,
        "workers": 2,
        "imgsz": 640,
        "optimizer": "AdamW",
        "lr": 0.00001,
        "AMP": False,
        "wandb": "disabled via WANDB_MODE=disabled",
        "NNCF API": "nncf.torch.create_compressed_model (never nncf.quantize)",
        "range-init method": "QuantizationController.init_range(PTRangeInitParams)",
        "train status": False,
        "validation status": False,
        "checkpoint status": False,
        "reload status": False,
        "notes": [
            "Smoke only: one epoch; no PTQ, OpenVINO export, test split, DDP, multi-GPU, cache=ram, or final QAT run.",
            "The generated data YAML contains only train and val keys; it intentionally has no test key.",
        ],
    }
    trainer: QATSmokeDetectionTrainer | None = None
    try:
        require_project_root(ROOT)
        source_checkpoint = (ROOT / CHECKPOINT_RELATIVE).resolve()
        if not source_checkpoint.is_file():
            raise SmokeStop(f"FP32 checkpoint not found: {source_checkpoint}")
        device, gpu_name = require_cuda()
        train_images = resolve_images(args.train_images, "train")
        val_images = resolve_images(args.val_images, "validation")
        names = load_names(ROOT)
        if len(names) != 3:
            raise SmokeStop(f"Expected nc=3 from Phase 5A, found {len(names)} classes in data/dataset/data.yaml.")
        smoke_dir = ROOT / SMOKE_RELATIVE
        weights_dir = smoke_dir / "weights"
        data_yaml = smoke_dir / "data_train_val.yaml"
        write_smoke_data_yaml(data_yaml, train_images, val_images, names)

        manifest.update(
            {
                "source FP32 SHA256": sha256_file(source_checkpoint),
                "GPU": {"logical_device": "cuda:0", "name": gpu_name},
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "train_images": str(train_images),
                "validation_images": str(val_images),
                "data_yaml": str(data_yaml),
                "range-init sample count": args.range_init_samples,
            }
        )
        print(f"Source checkpoint: {source_checkpoint}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Logical GPU (device=0): {gpu_name}")
        print(f"torch: {torch.__version__}; ultralytics: {ultralytics.__version__}; nncf: {nncf.__version__}")
        print("QAT lr0: 0.00001; AMP: False; NNCF API: nncf.torch.create_compressed_model")

        overrides = {
            "task": "detect",
            "mode": "train",
            "model": str(source_checkpoint),
            "data": str(data_yaml),
            "epochs": 1,
            "fraction": 0.10,
            "batch": 8,
            "workers": 2,
            "imgsz": 640,
            "device": "0",
            "seed": 42,
            "optimizer": "AdamW",
            "lr0": 0.00001,
            "amp": False,
            "cache": False,
            "val": True,
            "plots": False,
            "save": True,
            "pretrained": False,
            "resume": False,
            "project": str(smoke_dir),
            "name": ".",
            "exist_ok": True,
            "save_dir": str(smoke_dir),
        }
        trainer = QATSmokeDetectionTrainer(
            overrides=overrides,
            source_checkpoint=source_checkpoint,
            train_images=train_images,
            names=names,
            range_init_batch=args.range_init_batch,
            range_init_samples=args.range_init_samples,
        )
        if trainer.world_size != 1 or trainer.ddp:
            raise SmokeStop("Smoke trainer resolved a DDP/multi-GPU configuration; only one visible GPU is allowed.")
        trainer.train()
        fake_calls = trainer.remove_quantizer_hooks()
        manifest.update(
            {
                "quantizer count before training": trainer.quantizer_count_before_training,
                "fake quantization enabled before training": trainer.fake_quant_enabled_before_training,
                "fake quant call count": fake_calls,
                "range-init": trainer.range_init_report,
                "train status": True,
                "validation status": bool(trainer.metrics),
                "checkpoint status": trainer.qat_checkpoint_path is not None and trainer.qat_checkpoint_path.is_file(),
                "checkpoint": str(trainer.qat_checkpoint_path) if trainer.qat_checkpoint_path else None,
                "best_checkpoint": str(trainer.qat_best_checkpoint_path) if trainer.qat_best_checkpoint_path else None,
            }
        )
        if not manifest["checkpoint status"]:
            raise SmokeStop("The one-epoch trainer did not produce the required NNCF-aware checkpoint.")
        if fake_calls < 1:
            raise SmokeStop("No fake-quant calls occurred during the trainer lifecycle.")

        metrics = dict(trainer.metrics or {})
        manifest.update(
            {
                "precision": metric_value(metrics, "precision"),
                "recall": metric_value(metrics, "recall"),
                "mAP50": metric_value(metrics, "map50"),
                "mAP50-95": metric_value(metrics, "map50-95") or metric_value(metrics, "map"),
                "validation_metrics": metrics,
            }
        )
        reload_report = reload_and_validate(
            trainer.qat_checkpoint_path,
            source_checkpoint,
            device,
            trainer,
            train_images,
            val_images,
            names,
            smoke_dir,
        )
        manifest["reload"] = reload_report
        manifest["quantizer count after reload"] = reload_report["quantizer_count"]
        manifest["reload status"] = reload_report["status"]
        manifest["status"] = "passed"
        print("QAT DetectionTrainer smoke passed; explicit NNCF reload verification passed.")
    except (SmokeStop, ProbeStop) as error:
        manifest["status"] = "stopped"
        manifest["error"] = f"{type(error).__name__}: {error}"
        print(f"QAT smoke stopped: {manifest['error']}", file=sys.stderr)
        return_code = 2
    except Exception as error:  # Preserve a diagnostic manifest for host-only failures.
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        print(f"QAT smoke failed: {manifest['error']}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        if trainer is not None:
            trainer.remove_quantizer_hooks()
        write_manifest(manifest_path, manifest)
        print(f"Manifest: {manifest_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
