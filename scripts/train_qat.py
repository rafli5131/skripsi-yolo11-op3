"""Run or resume the fixed five-epoch NNCF QAT fine-tuning experiment.

This script starts a new run only from the immutable FP32 checkpoint, inserts
NNCF PyTorch fake quantizers with ``create_compressed_model()``, and uses the
installed Ultralytics DetectionTrainer lifecycle.  ``nncf.quantize`` (PTQ),
OpenVINO export, and the dataset test split are never used.

Run on the CUDA SSH host from the repository root.  CUDA_VISIBLE_DEVICES is
supplied by the shell/.env before torch is imported; this script never edits it.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path.cwd().resolve()
load_dotenv(ROOT / ".env", override=False)

import nncf
import openvino
import torch
import torchvision
import ultralytics
import yaml
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, load_state
from ultralytics import YOLO, settings
from ultralytics.utils.torch_utils import unwrap_model

# Reuse the Phase 5B implementation rather than recreating another QAT path.
from smoke_train_qat import (  # noqa: E402
    QATSmokeDetectionTrainer,
    SmokeStop,
    initialize_ranges,
    make_nncf_structure_config,
    metric_value,
    quantizers,
    quantizers_enabled,
    reload_and_validate,
    resolve_images,
    sha256_file,
    tensor_sha256,
    write_smoke_data_yaml,
    write_manifest,
)
from probe_qat_compatibility import (  # noqa: E402
    ProbeStop,
    attach_ultralytics_training_args,
    load_names,
    require_cuda,
    require_project_root,
    restore_ultralytics_trainer_trainability,
)


SOURCE_CHECKPOINT = Path("artifacts/checkpoints/fp32/best.pt")
RUN_DIR = Path("experiments/qat/qat_final")
ARTIFACT_DIR = Path("artifacts/checkpoints/qat")
TRAIN_IMAGES = Path("data/raw/advance_vision_human/train/images")
VAL_IMAGES = Path("data/raw/advance_vision_human/valid/images")
RUN_NAME = "qat-yolo11n-5ep"
RANGE_INIT_SAMPLES = 16
RANGE_INIT_BATCH = 2
EPOCHS = 5
BATCH = 16
WORKERS = 2
IMGSZ = 640
SEED = 42
LR0 = 0.00001
LRF = 1.0
WARMUP_EPOCHS = 0.0
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9


class FinalQATStop(RuntimeError):
    """A clear, recorded stop for an invalid final-QAT state."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume only from weights/qat_last.pt.")
    parser.add_argument(
        "--reload-only",
        action="store_true",
        help="Verify existing qat_best.pt and freeze artifacts without running training.",
    )
    parser.add_argument("--enable-wandb", action="store_true", help="Enable opt-in Weights & Biases logging.")
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT") or "skripsi-yolo11-qat-openvino",
        help="W&B project when --enable-wandb is supplied.",
    )
    parser.add_argument(
        "--wandb-name",
        default=os.environ.get("WANDB_NAME") or None,
        help="W&B name when --enable-wandb is supplied; defaults to WANDB_NAME or qat-yolo11n-5ep.",
    )
    return parser.parse_args()


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def read_resume_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalQATStop(f"--resume requires the NNCF QAT checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "format",
        "epoch",
        "model_state_dict",
        "compression_state",
        "nncf_config",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "ema_state_dict",
        "best_fitness",
        "parameter_fingerprint",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise FinalQATStop(f"Resume checkpoint is incomplete; missing: {missing}")
    if payload["format"] != "nncf_qat_final_v1":
        raise FinalQATStop(f"Resume checkpoint has unsupported format: {payload['format']!r}")
    return payload


class FinalQATDetectionTrainer(QATSmokeDetectionTrainer):
    """The Phase 5B trainer plus strict NNCF-aware resume/checkpoint state."""

    def __init__(self, *args: Any, resume_payload: dict[str, Any] | None, **kwargs: Any) -> None:
        self.final_resume_payload = resume_payload
        self.best_epoch = -1
        self.best_metrics: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):  # type: ignore[override]
        if self.final_resume_payload is None:
            return super().get_model(cfg, weights, verbose)

        del cfg, weights, verbose
        if self.qat_model_prepared:
            raise FinalQATStop("Refusing to create NNCF compressed structure more than once during resume.")
        payload = self.final_resume_payload
        fp32_model = YOLO(str(self.source_checkpoint)).model
        if int(getattr(fp32_model, "nc", -1)) != int(self.data["nc"]):
            raise FinalQATStop("FP32 source class count does not match final QAT data.")
        attach_ultralytics_training_args(fp32_model, ROOT)
        restore_ultralytics_trainer_trainability(fp32_model)
        self.nncf_structure_config = NNCFConfig.from_dict(payload["nncf_config"])
        controller, qat_model = create_compressed_model(
            fp32_model,
            self.nncf_structure_config,
            compression_state=payload["compression_state"],
            dump_graphs=False,
        )
        qat_model.to(self.device)
        loaded = load_state(qat_model, payload["model_state_dict"], is_resume=True)
        items = quantizers(qat_model)
        if not items or not quantizers_enabled(items):
            raise FinalQATStop("Resume reconstruction did not retain active fake quantizers.")
        self.compression_ctrl = controller
        self.qat_model_prepared = True
        self.quantizer_count_before_training = len(items)
        self.fake_quant_enabled_before_training = True
        self.range_init_report = {
            "method": "restored from NNCF compression_state; range initialization is not repeated on resume",
            "loaded_state_entries": int(loaded),
        }
        self.parameter_fingerprint = dict(payload["parameter_fingerprint"])
        self.fake_quant_calls, self._fake_quant_hooks = self._install_resume_quantizer_counter(items)
        print(
            f"Resumed NNCF QAT model: quantizers={len(items)}; loaded_state_entries={loaded}; "
            "fake_quant_enabled=True",
            flush=True,
        )
        return qat_model

    @staticmethod
    def _install_resume_quantizer_counter(items):
        # Keep this local to avoid modifying Phase 5B's already verified class.
        calls = {id(item): 0 for item in items}
        hooks = [
            item.register_forward_hook(
                lambda _module, _inputs, _output, item_id=id(item): calls.__setitem__(item_id, calls[item_id] + 1)
            )
            for item in items
        ]
        return calls, hooks

    def resume_training(self, ckpt) -> None:  # type: ignore[override]
        """Restore optimizer/scheduler/EMA after BaseTrainer has built them."""
        del ckpt  # BaseTrainer's ordinary FP32 checkpoint is intentionally ignored.
        payload = self.final_resume_payload
        if payload is None:
            return
        start_epoch = int(payload["epoch"]) + 1
        if not 0 < start_epoch < self.epochs:
            raise FinalQATStop(
                f"Final QAT checkpoint is already at epoch {start_epoch}/{self.epochs}; "
                "--resume must not silently restart from FP32."
            )
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.scaler.load_state_dict(payload["scaler_state_dict"])
        self.ema.ema.load_state_dict(payload["ema_state_dict"], strict=True)
        self.ema.updates = int(payload.get("ema_updates", 0))
        self.best_fitness = payload["best_fitness"]
        self.best_epoch = int(payload.get("best_epoch", -1))
        self.best_metrics = dict(payload.get("best_metrics", {}))
        self.start_epoch = start_epoch
        print(f"Resuming final QAT from epoch {start_epoch + 1}/{self.epochs}.")

    def save_model(self) -> bool:  # type: ignore[override]
        """Persist every state needed by NNCF and normal optimizer resume."""
        if self.compression_ctrl is None or self.nncf_structure_config is None:
            raise FinalQATStop("Cannot save final QAT before NNCF preparation.")
        model = unwrap_model(self.model)
        state_dict = cpu_state_dict(model)
        fingerprint = dict(self.parameter_fingerprint or {})
        if fingerprint.get("name") not in state_dict:
            raise FinalQATStop("QAT fingerprint parameter is absent from the checkpoint state dictionary.")
        fingerprint["after"] = tensor_sha256(state_dict[fingerprint["name"]])
        is_best = self.best_fitness == self.fitness
        if is_best:
            self.best_epoch = int(self.epoch) + 1
            self.best_metrics = dict(self.metrics or {})
        payload = {
            "format": "nncf_qat_final_v1",
            "epoch": int(self.epoch),
            "source_fp32_checkpoint": str(self.source_checkpoint),
            "model_state_dict": state_dict,
            "compression_state": copy.deepcopy(self.compression_ctrl.get_compression_state()),
            "nncf_config": copy.deepcopy(dict(self.nncf_structure_config)),
            "optimizer_state_dict": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler_state_dict": copy.deepcopy(self.scheduler.state_dict()),
            "scaler_state_dict": copy.deepcopy(self.scaler.state_dict()),
            "ema_state_dict": cpu_state_dict(self.ema.ema),
            "ema_updates": int(self.ema.updates),
            "best_fitness": self.best_fitness,
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "train_args": vars(self.args).copy(),
            "metrics": dict(self.metrics or {}),
            "parameter_fingerprint": fingerprint,
        }
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.qat_checkpoint_path = self.wdir / "qat_last.pt"
        torch.save(payload, self.qat_checkpoint_path)
        if is_best:
            self.qat_best_checkpoint_path = self.wdir / "qat_best.pt"
            torch.save(payload, self.qat_best_checkpoint_path)
        elif self.qat_best_checkpoint_path is None:
            candidate = self.wdir / "qat_best.pt"
            self.qat_best_checkpoint_path = candidate if candidate.is_file() else None
        return True


def initialize_wandb(enabled: bool, project: str, name: str, config: dict[str, Any]):
    if not enabled:
        return None, None
    # smoke_train_qat disables W&B for its own run at import time.  Re-enable
    # only for this explicit CLI request, without printing any credential.
    os.environ["WANDB_MODE"] = "online"
    try:
        import wandb
    except ImportError as error:
        raise FinalQATStop("--enable-wandb requires the project wandb dependency. Run uv sync first.") from error
    run = wandb.init(project=project, name=name, config=config, job_type="train")

    def on_fit_epoch_end(trainer: Any) -> None:
        metrics: dict[str, float | int] = {
            "epoch": int(trainer.epoch) + 1,
            "qat/quantizer_count": int(trainer.quantizer_count_before_training),
        }
        for source in (getattr(trainer, "tloss", {}), getattr(trainer, "metrics", {}), getattr(trainer, "lr", {})):
            for key, value in source.items():
                try:
                    metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        run.log(metrics, step=int(trainer.epoch) + 1)

    return run, on_fit_epoch_end


def checkpoint_format_file(run_dir: Path) -> Path:
    path = run_dir / "nncf_checkpoint_format.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "format": "nncf_qat_final_v1",
                "reconstruction": [
                    "load FP32 YOLO architecture from artifacts/checkpoints/fp32/best.pt",
                    "create_compressed_model(fp32_model, NNCFConfig.from_dict(nncf_config), compression_state=...) ",
                    "nncf.torch.load_state(qat_model, model_state_dict, is_resume=True)",
                ],
                "embedded_checkpoint_fields": ["compression_state", "nncf_config", "model_state_dict"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def freeze_artifacts(run_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Copy immutable final deliverables only after reload verification succeeds."""
    artifact_dir = ROOT / ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        run_dir / "weights" / "qat_best.pt",
        run_dir / "weights" / "qat_last.pt",
        run_dir / "nncf_structure_config.json",
        run_dir / "nncf_checkpoint_format.json",
        run_dir / "data_train_val.yaml",
        run_dir / "results.csv",
        manifest_path,
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FinalQATStop(f"Artifact freeze aborted; required files are missing: {missing}")
    copied = []
    for source in sources:
        target = artifact_dir / source.name
        shutil.copy2(source, target)
        copied.append(target)
    hashes = {path.name: sha256_file(path) for path in copied if path.suffix in {".pt", ".json"}}
    (artifact_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())), encoding="utf-8"
    )
    return {"directory": str(artifact_dir), "files": [str(path) for path in copied], "sha256": hashes}


def reload_only() -> int:
    """Complete reload/freeze for a run whose five epochs already finished."""
    run_dir = ROOT / RUN_DIR
    manifest_path = run_dir / "manifest.json"
    best_path = run_dir / "weights" / "qat_best.pt"
    try:
        require_project_root(ROOT)
        if not manifest_path.is_file() or not best_path.is_file():
            raise FinalQATStop("--reload-only requires an existing final manifest and weights/qat_best.pt.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_checkpoint = (ROOT / SOURCE_CHECKPOINT).resolve()
        device, gpu_name = require_cuda()
        train_images = resolve_images(TRAIN_IMAGES, "train")
        val_images = resolve_images(VAL_IMAGES, "validation")
        names = load_names(ROOT)
        print(f"Reload-only verification: {best_path}")
        reload_report = reload_and_validate(
            best_path, source_checkpoint, device, None, train_images, val_images, names, run_dir
        )
        if reload_report["quantizer_count"] != 184:
            raise FinalQATStop(f"Expected 184 reload quantizers, found {reload_report['quantizer_count']}.")
        manifest.update(
            {
                "checkpoint reload verification status": True,
                "reload verification": reload_report,
                "quantizer count after reload": reload_report["quantizer_count"],
                "status": "passed",
                "reload_verified_at": datetime.now(timezone.utc).isoformat(),
                "GPU": {"logical_device": "cuda:0", "name": gpu_name},
            }
        )
        manifest.pop("error", None)
        write_manifest(manifest_path, manifest)
        manifest["artifact freeze"] = freeze_artifacts(run_dir, manifest_path)
        write_manifest(manifest_path, manifest)
        shutil.copy2(manifest_path, ROOT / ARTIFACT_DIR / manifest_path.name)
        print("Final QAT reload verification and artifact freeze passed (no training run).")
        print(f"Manifest: {manifest_path}")
        return 0
    except (FinalQATStop, SmokeStop, ProbeStop) as error:
        message = f"{type(error).__name__}: {error}"
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        manifest.update({"status": "failed", "checkpoint reload verification status": False, "error": message})
        write_manifest(manifest_path, manifest)
    finally:
        print(f"Final QAT reload-only failed: {message}", file=sys.stderr)
        print(f"Manifest: {manifest_path}")
    return 1


def main() -> int:
    args = parse_args()
    if args.reload_only:
        return reload_only()
    run_dir = ROOT / RUN_DIR
    manifest_path = run_dir / "manifest.json"
    wandb_run = None
    trainer: FinalQATDetectionTrainer | None = None
    manifest: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "started"}
    try:
        require_project_root(ROOT)
        source_checkpoint = (ROOT / SOURCE_CHECKPOINT).resolve()
        if not source_checkpoint.is_file():
            raise FinalQATStop(f"FP32 source checkpoint not found: {source_checkpoint}")
        final_last = run_dir / "weights" / "qat_last.pt"
        if final_last.exists() and not args.resume:
            raise FinalQATStop(f"Existing final QAT checkpoint found: {final_last}. Use --resume; do not overwrite it.")
        resume_payload = read_resume_payload(final_last) if args.resume else None
        if resume_payload is not None and Path(resume_payload.get("source_fp32_checkpoint", "")).resolve() != source_checkpoint:
            raise FinalQATStop("Resume checkpoint was not created from this required FP32 source checkpoint.")
        device, gpu_name = require_cuda()
        train_images = resolve_images(TRAIN_IMAGES, "train")
        val_images = resolve_images(VAL_IMAGES, "validation")
        names = load_names(ROOT)
        expected_names = {0: "ball", 1: "gawang", 2: "robot"}
        if names != expected_names:
            raise FinalQATStop(f"Expected class mapping {expected_names}, got {names}.")

        # Deliberately do not use data/dataset/data.yaml directly: it declares
        # test.  This run-local YAML has only full train and validation entries.
        data_yaml = run_dir / "data_train_val.yaml"
        write_smoke_data_yaml(data_yaml, train_images, val_images, names)
        checkpoint_format = checkpoint_format_file(run_dir)
        base_config = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
        settings.update({"wandb": False})  # manual W&B below is the only opt-in path.

        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "started",
            "source FP32 checkpoint": str(source_checkpoint),
            "source FP32 SHA256": sha256_file(source_checkpoint),
            "versions": {
                "Python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "torch CUDA": torch.version.cuda,
                "ultralytics": ultralytics.__version__,
                "nncf": nncf.__version__,
                "openvino": openvino.__version__,
            },
            "GPU": {"logical_device": "cuda:0", "name": gpu_name},
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "dataset YAML": str(data_yaml),
            "dataset workspace": base_config["dataset"]["workspace"],
            "dataset project": base_config["dataset"]["project"],
            "dataset version": base_config["dataset"]["version"],
            "class mapping": names,
            "epochs": EPOCHS,
            "batch": BATCH,
            "workers": WORKERS,
            "imgsz": IMGSZ,
            "seed": SEED,
            "optimizer": "AdamW",
            "learning rate": LR0,
            "lrf": LRF,
            "warmup_epochs": WARMUP_EPOCHS,
            "weight decay": WEIGHT_DECAY,
            "momentum": MOMENTUM,
            "AMP": False,
            "cache": False,
            "QAT API": "nncf.torch.create_compressed_model (never nncf.quantize)",
            "range initialization method": "QuantizationController.init_range(PTRangeInitParams)",
            "range initialization sample count": RANGE_INIT_SAMPLES,
            "expected quantizer count": 184,
            "resume": bool(args.resume),
            "W&B": {"enabled": bool(args.enable_wandb), "project": args.wandb_project if args.enable_wandb else None},
            "notes": [
                "Final fixed five-epoch QAT only: no PTQ, OpenVINO export, test split, DDP, multi-GPU, cache=ram, or architecture changes.",
                "Range initialization receives a deterministic first-16-images TRAIN-only subset.",
            ],
        }
        write_manifest(manifest_path, manifest)
        print(f"Source FP32 checkpoint: {source_checkpoint}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Logical GPU (device=0): {gpu_name}")
        print(
            f"Final QAT: epochs={EPOCHS}, batch={BATCH}, lr0={LR0}, lrf={LRF}, "
            f"warmup_epochs={WARMUP_EPOCHS}, AMP=False, fraction=1.0"
        )

        overrides = {
            "task": "detect", "mode": "train", "model": str(source_checkpoint), "data": str(data_yaml),
            "epochs": EPOCHS, "fraction": 1.0, "batch": BATCH, "workers": WORKERS, "imgsz": IMGSZ,
            "device": "0", "seed": SEED, "optimizer": "AdamW", "lr0": LR0, "lrf": LRF,
            "warmup_epochs": WARMUP_EPOCHS, "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY, "amp": False, "cache": False, "val": True, "plots": False,
            "save": True, "pretrained": False, "resume": False, "project": str(run_dir), "name": ".",
            "exist_ok": True, "save_dir": str(run_dir),
        }
        trainer = FinalQATDetectionTrainer(
            overrides=overrides,
            source_checkpoint=source_checkpoint,
            train_images=train_images,
            names=names,
            range_init_batch=RANGE_INIT_BATCH,
            range_init_samples=RANGE_INIT_SAMPLES,
            resume_payload=resume_payload,
        )
        if trainer.world_size != 1 or trainer.ddp:
            raise FinalQATStop("Final QAT resolved DDP/multi-GPU; exactly one visible GPU is required.")
        wandb_name = args.wandb_name or RUN_NAME
        wandb_run, wandb_callback = initialize_wandb(args.enable_wandb, args.wandb_project, wandb_name, manifest)
        if wandb_callback is not None:
            trainer.add_callback("on_fit_epoch_end", wandb_callback)
            manifest["W&B"].update({"name": wandb_name, "run ID": wandb_run.id})
            write_manifest(manifest_path, manifest)

        trainer.train()
        fake_calls = trainer.remove_quantizer_hooks()
        last_path = trainer.qat_checkpoint_path or (run_dir / "weights" / "qat_last.pt")
        best_path = trainer.qat_best_checkpoint_path or (run_dir / "weights" / "qat_best.pt")
        if not last_path.is_file() or not best_path.is_file():
            raise FinalQATStop("Final QAT did not produce both qat_last.pt and validation-selected qat_best.pt.")
        quantizer_count_after_training = len(quantizers(unwrap_model(trainer.model)))
        if quantizer_count_after_training != trainer.quantizer_count_before_training:
            raise FinalQATStop(
                "Quantizer count changed during QAT training: "
                f"{trainer.quantizer_count_before_training} -> {quantizer_count_after_training}."
            )
        config_path = run_dir / "nncf_structure_config.json"
        config_path.write_text(json.dumps(dict(trainer.nncf_structure_config), indent=2) + "\n", encoding="utf-8")

        final_metrics = dict(trainer.metrics or {})
        manifest.update(
            {
                "quantizer count before training": trainer.quantizer_count_before_training,
                "quantizer count after training": quantizer_count_after_training,
                "fake quant call count": fake_calls,
                "range initialization": trainer.range_init_report,
                "best epoch": trainer.best_epoch,
                "best precision": metric_value(trainer.best_metrics, "precision"),
                "best recall": metric_value(trainer.best_metrics, "recall"),
                "best mAP50": metric_value(trainer.best_metrics, "map50"),
                "best mAP50-95": metric_value(trainer.best_metrics, "map50-95") or metric_value(trainer.best_metrics, "map"),
                "final epoch metrics": final_metrics,
                "checkpoint paths": {"qat_last": str(last_path), "qat_best": str(best_path)},
            }
        )
        reload_report = reload_and_validate(best_path, source_checkpoint, device, trainer, train_images, val_images, names, run_dir)
        if reload_report["quantizer_count"] != trainer.quantizer_count_before_training:
            raise FinalQATStop(
                "Reloaded checkpoint quantizer count differs from prepared model: "
                f"{reload_report['quantizer_count']} != {trainer.quantizer_count_before_training}."
            )
        manifest["checkpoint reload verification status"] = reload_report["status"]
        manifest["reload verification"] = reload_report
        manifest["status"] = "passed"
        # Write config/manifest before freeze.  The final write and copy below
        # ensure the artifact copy contains the completed freeze metadata too.
        write_manifest(manifest_path, manifest)
        manifest["artifact freeze"] = freeze_artifacts(run_dir, manifest_path)
        write_manifest(manifest_path, manifest)
        shutil.copy2(manifest_path, ROOT / ARTIFACT_DIR / manifest_path.name)
        print("Final QAT completed; NNCF-aware best-checkpoint reload and artifact freeze passed.")
    except (FinalQATStop, SmokeStop, ProbeStop) as error:
        manifest.update({"status": "stopped", "error": f"{type(error).__name__}: {error}"})
        print(f"Final QAT stopped: {manifest['error']}", file=sys.stderr)
        return_code = 2
    except Exception as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        print(f"Final QAT failed: {manifest['error']}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        if trainer is not None:
            trainer.remove_quantizer_hooks()
        write_manifest(manifest_path, manifest)
        if wandb_run is not None:
            wandb_run.finish()
        print(f"Manifest: {manifest_path}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
