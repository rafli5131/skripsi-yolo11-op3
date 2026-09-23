"""Tune true NNCF-QAT candidates on validation data and promote the best one.

This script intentionally does NOT use PTQ metrics and does NOT read the TEST
split for model selection. Every candidate executes the repository's existing
true-QAT training path in ``scripts/train_qat.py`` with different QAT
fine-tuning hyperparameters. The winner is selected by full-validation
mAP50-95, with mAP50/precision/recall as deterministic tie breakers.

Run from the repository root on the CUDA host, for example:

    export CUDA_VISIBLE_DEVICES=1
    env -u PYTHONPATH -u VIRTUAL_ENV uv run python scripts/tune_qat_best.py

After selection, the winning candidate artifacts are copied to the canonical
``artifacts/checkpoints/qat`` directory. Only then should the frozen TEST split
be evaluated once with ``scripts/evaluate_final_test.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/qat_best_search.yaml"
SEARCH_ROOT = ROOT / "experiments/qat/qat_best_search"
RUN_ROOT = SEARCH_ROOT / "runs"
SEARCH_ARTIFACT_ROOT = ROOT / "artifacts/checkpoints/qat_search"
CANONICAL_QAT_DIR = ROOT / "artifacts/checkpoints/qat"
SELECTION_MANIFEST = SEARCH_ROOT / "selection_manifest.json"


class TuneStop(RuntimeError):
    """Controlled stop for invalid QAT tuning state."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rerun candidate directories even if a passed manifest already exists.",
    )
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Select the best candidate but do not copy it to artifacts/checkpoints/qat.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the search configuration and print candidates without training.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sha256s(directory: Path) -> None:
    tracked = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix in {".pt", ".json"}
    )
    (directory / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in tracked),
        encoding="utf-8",
    )


def load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (ROOT / resolved).resolve()
    if not resolved.is_file():
        raise TuneStop(f"QAT search config not found: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TuneStop("QAT search config must be a YAML mapping.")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TuneStop("QAT search config must contain a non-empty candidates list.")
    return payload


def normalized_candidate(common: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(common)
    candidate.update(raw)
    required = {
        "name",
        "epochs",
        "lr0",
        "lrf",
        "warmup_epochs",
        "range_init_samples",
        "range_init_batch",
        "batch",
        "workers",
        "imgsz",
        "seed",
        "weight_decay",
        "momentum",
    }
    missing = sorted(required.difference(candidate))
    if missing:
        raise TuneStop(f"QAT candidate missing required fields: {missing}")
    name = str(candidate["name"])
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise TuneStop(f"Unsafe candidate name: {name!r}")
    candidate["name"] = name
    candidate["epochs"] = int(candidate["epochs"])
    candidate["batch"] = int(candidate["batch"])
    candidate["workers"] = int(candidate["workers"])
    candidate["imgsz"] = int(candidate["imgsz"])
    candidate["seed"] = int(candidate["seed"])
    candidate["range_init_samples"] = int(candidate["range_init_samples"])
    candidate["range_init_batch"] = int(candidate["range_init_batch"])
    for key in ("lr0", "lrf", "warmup_epochs", "weight_decay", "momentum"):
        candidate[key] = float(candidate[key])
    if candidate["epochs"] < 1:
        raise TuneStop(f"{name}: epochs must be >= 1")
    if candidate["range_init_samples"] < candidate["range_init_batch"]:
        raise TuneStop(f"{name}: range_init_samples must be >= range_init_batch")
    if candidate["imgsz"] != 640:
        raise TuneStop(f"{name}: imgsz must stay 640 for this experiment")
    return candidate


def candidate_paths(name: str) -> tuple[Path, Path, Path]:
    run_dir = RUN_ROOT / name
    artifact_dir = SEARCH_ARTIFACT_ROOT / name
    manifest = run_dir / "manifest.json"
    return run_dir, artifact_dir, manifest


def passed_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if data.get("status") == "passed" else None


def child_code(candidate: dict[str, Any], resume: bool) -> str:
    name = candidate["name"]
    run_rel = Path("experiments/qat/qat_best_search/runs") / name
    artifact_rel = Path("artifacts/checkpoints/qat_search") / name
    values = {
        "RUN_DIR": str(run_rel),
        "ARTIFACT_DIR": str(artifact_rel),
        "RUN_NAME": name,
        "EPOCHS": candidate["epochs"],
        "BATCH": candidate["batch"],
        "WORKERS": candidate["workers"],
        "IMGSZ": candidate["imgsz"],
        "SEED": candidate["seed"],
        "LR0": candidate["lr0"],
        "LRF": candidate["lrf"],
        "WARMUP_EPOCHS": candidate["warmup_epochs"],
        "WEIGHT_DECAY": candidate["weight_decay"],
        "MOMENTUM": candidate["momentum"],
        "RANGE_INIT_SAMPLES": candidate["range_init_samples"],
        "RANGE_INIT_BATCH": candidate["range_init_batch"],
    }
    assignments = "\n".join(f"q.{key} = {value!r}" for key, value in values.items())
    argv = ["train_qat.py", "--resume"] if resume else ["train_qat.py"]
    return f"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / 'scripts'))
import train_qat as q
{assignments}
sys.argv = {argv!r}
raise SystemExit(q.main())
"""


def run_candidate(candidate: dict[str, Any], force: bool) -> dict[str, Any]:
    name = candidate["name"]
    run_dir, artifact_dir, manifest_path = candidate_paths(name)

    existing = passed_manifest(manifest_path)
    if existing is not None and not force:
        print(f"[SKIP] {name}: existing passed manifest")
        annotate_manifest(manifest_path, candidate)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    if force:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(artifact_dir, ignore_errors=True)

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    resume = (run_dir / "weights" / "qat_last.pt").is_file() and not force

    action = "RESUME" if resume else "RUN"
    print(
        f"[{action}] {name}: epochs={candidate['epochs']} lr0={candidate['lr0']} "
        f"lrf={candidate['lrf']} range_init_samples={candidate['range_init_samples']}"
    )
    completed = subprocess.run(
        [sys.executable, "-c", child_code(candidate, resume)],
        cwd=ROOT,
        env=os.environ.copy(),
        check=False,
    )
    if completed.returncode != 0:
        raise TuneStop(f"Candidate {name} failed with exit code {completed.returncode}.")
    if not manifest_path.is_file():
        raise TuneStop(f"Candidate {name} completed without manifest: {manifest_path}")

    annotate_manifest(manifest_path, candidate)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise TuneStop(f"Candidate {name} manifest status is not passed.")
    return manifest


def annotate_manifest(manifest_path: Path, candidate: dict[str, Any]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["QAT tuning candidate"] = candidate
    manifest["selection eligibility"] = "validation-only true-QAT candidate"
    manifest["TEST split used for selection"] = False
    manifest["PTQ metrics used for selection"] = False
    notes = [
        note
        for note in manifest.get("notes", [])
        if "fixed five-epoch" not in str(note).lower()
    ]
    notes.append(
        "Validation-driven QAT tuning candidate; epoch count and range initialization are defined by configs/qat_best_search.yaml."
    )
    manifest["notes"] = notes
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def metric(manifest: dict[str, Any], key: str) -> float:
    aliases = {
        "mAP50-95": ("best mAP50-95",),
        "mAP50": ("best mAP50",),
        "precision": ("best precision",),
        "recall": ("best recall",),
    }
    for alias in aliases.get(key, (key,)):
        value = manifest.get(alias)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float("-inf")


def select_best(
    results: list[tuple[dict[str, Any], dict[str, Any]]],
    selection_metric: str,
    secondary_metrics: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = [selection_metric, *secondary_metrics]

    def score(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, ...]:
        _, manifest = item
        return tuple(metric(manifest, key) for key in keys)

    winner = max(results, key=score)
    if score(winner)[0] == float("-inf"):
        raise TuneStop(f"No candidate exposes selection metric {selection_metric!r}.")
    return winner


def promote(candidate: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    name = candidate["name"]
    _, source_artifacts, run_manifest = candidate_paths(name)
    artifact_files = [
        "qat_best.pt",
        "qat_last.pt",
        "nncf_structure_config.json",
        "nncf_checkpoint_format.json",
        "data_train_val.yaml",
        "results.csv",
    ]
    missing = [filename for filename in artifact_files if not (source_artifacts / filename).is_file()]
    if not run_manifest.is_file():
        missing.append("run manifest.json")
    if missing:
        raise TuneStop(f"Cannot promote {name}; missing artifacts: {missing}")

    CANONICAL_QAT_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for filename in artifact_files:
        source = source_artifacts / filename
        destination = CANONICAL_QAT_DIR / filename
        shutil.copy2(source, destination)
        copied.append(str(destination))

    canonical_manifest = CANONICAL_QAT_DIR / "manifest.json"
    shutil.copy2(run_manifest, canonical_manifest)
    copied.append(str(canonical_manifest))

    promotion = {
        "promoted": True,
        "candidate": name,
        "source_artifact_dir": str(source_artifacts),
        "canonical_artifact_dir": str(CANONICAL_QAT_DIR),
        "copied": copied,
        "validation_metrics": {
            "mAP50-95": metric(manifest, "mAP50-95"),
            "mAP50": metric(manifest, "mAP50"),
            "precision": metric(manifest, "precision"),
            "recall": metric(manifest, "recall"),
        },
        "TEST split used for selection": False,
        "PTQ metrics used for selection": False,
    }
    return promotion


def write_selection_manifest(
    config_path: Path,
    selection_metric: str,
    secondary_metrics: list[str],
    results: list[tuple[dict[str, Any], dict[str, Any]]],
    winner: tuple[dict[str, Any], dict[str, Any]],
    promotion: dict[str, Any] | None,
) -> None:
    candidate_rows = []
    for candidate, manifest in results:
        candidate_rows.append(
            {
                "name": candidate["name"],
                "hyperparameters": candidate,
                "validation": {
                    "mAP50-95": metric(manifest, "mAP50-95"),
                    "mAP50": metric(manifest, "mAP50"),
                    "precision": metric(manifest, "precision"),
                    "recall": metric(manifest, "recall"),
                },
                "manifest": str(candidate_paths(candidate["name"])[2]),
            }
        )

    winner_candidate, winner_manifest = winner
    payload = {
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_method": "true NNCF-QAT validation-only model selection",
        "config": str(config_path),
        "selection_metric": selection_metric,
        "secondary_metrics": secondary_metrics,
        "TEST split used for selection": False,
        "PTQ metrics used for selection": False,
        "candidates": candidate_rows,
        "winner": {
            "name": winner_candidate["name"],
            "validation": {
                "mAP50-95": metric(winner_manifest, "mAP50-95"),
                "mAP50": metric(winner_manifest, "mAP50"),
                "precision": metric(winner_manifest, "precision"),
                "recall": metric(winner_manifest, "recall"),
            },
        },
        "promotion": promotion,
        "next_step": "Evaluate the frozen TEST split once with scripts/evaluate_final_test.py after model selection is complete.",
    }
    SELECTION_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if promotion:
        (CANONICAL_QAT_DIR / "selection_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_sha256s(CANONICAL_QAT_DIR)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    common = dict(config.get("common") or {})
    candidates = [normalized_candidate(common, item) for item in config["candidates"]]
    names = [candidate["name"] for candidate in candidates]
    if len(names) != len(set(names)):
        raise TuneStop("Candidate names must be unique.")

    selection_metric = str(config.get("selection_metric") or "mAP50-95")
    secondary_metrics = [str(value) for value in config.get("secondary_metrics") or []]

    print(f"True-QAT search: {len(candidates)} candidates")
    print(f"Selection: {selection_metric}; tie-breakers={secondary_metrics}")
    print("TEST used for selection: False")
    print("PTQ metrics used for selection: False")
    for candidate in candidates:
        print(
            f"  - {candidate['name']}: epochs={candidate['epochs']} lr0={candidate['lr0']} "
            f"lrf={candidate['lrf']} range_init_samples={candidate['range_init_samples']}"
        )

    if args.dry_run:
        return 0

    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for candidate in candidates:
        manifest = run_candidate(candidate, args.force)
        results.append((candidate, manifest))

    winner = select_best(results, selection_metric, secondary_metrics)
    winner_candidate, winner_manifest = winner
    print(
        f"[BEST] {winner_candidate['name']}: "
        f"mAP50-95={metric(winner_manifest, 'mAP50-95'):.6f}, "
        f"mAP50={metric(winner_manifest, 'mAP50'):.6f}"
    )

    promotion = None if args.no_promote else promote(winner_candidate, winner_manifest)
    write_selection_manifest(
        Path(args.config).resolve(),
        selection_metric,
        secondary_metrics,
        results,
        winner,
        promotion,
    )
    print(f"Selection manifest: {SELECTION_MANIFEST}")
    if promotion:
        print(f"Canonical QAT artifacts promoted to: {CANONICAL_QAT_DIR}")
    print("Next: run scripts/evaluate_final_test.py once on the frozen TEST split.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TuneStop as error:
        print(f"QAT tuning failed: {error}", file=sys.stderr)
        raise SystemExit(1)
