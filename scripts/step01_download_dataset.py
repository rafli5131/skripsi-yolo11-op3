"""Download the configured Roboflow YOLO11 detection dataset and normalize paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Keep shell credentials authoritative while accepting root .env defaults.
load_dotenv(Path.cwd() / ".env", override=False)

from roboflow import Roboflow


EXPECTED_NAMES = {0: "ball", 1: "gawang", 2: "robot"}
REQUIRED_SPLITS = ("train", "valid", "test")


def project_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("Run this script from the project root (directory containing pyproject.toml).")
    return root


def require_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required. Set it in the shell or in the project .env file.")
    return value


def normalize_names(raw_names: object) -> dict[int, str]:
    if isinstance(raw_names, list):
        names = {index: str(value) for index, value in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        try:
            names = {int(key): str(value) for key, value in raw_names.items()}
        except (TypeError, ValueError) as error:
            raise RuntimeError("Source data.yaml names keys must be integer class IDs.") from error
    else:
        raise RuntimeError("Source data.yaml does not contain a valid names mapping.")

    if names != EXPECTED_NAMES:
        raise RuntimeError(
            f"Unexpected source class mapping: {names!r}. Expected exactly: {EXPECTED_NAMES!r}. "
            "No class remapping was performed."
        )
    return names


def locate_dataset_root(destination: Path, downloaded_location: object) -> Path:
    candidates = []
    if downloaded_location:
        candidates.append(Path(str(downloaded_location)))
    candidates.extend((destination, destination / "advance_vision_human"))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if all((candidate / split / leaf).is_dir() for split in REQUIRED_SPLITS for leaf in ("images", "labels")):
            return candidate

    checked = "\n".join(f"- {path}" for path in seen)
    raise RuntimeError(
        "Downloaded export does not contain the expected train/valid/test image and label directories. "
        f"Checked:\n{checked}"
    )


def main() -> int:
    root = project_root()
    api_key = require_environment("ROBOFLOW_API_KEY")
    version_id = require_environment("ROBOFLOW_VERSION")

    config = yaml.safe_load((root / "configs" / "base.yaml").read_text(encoding="utf-8"))
    dataset_config = config["dataset"]
    destination = root / "data" / "raw" / "advance_vision_human"

    print(f"Downloading Roboflow dataset to: {destination}")
    roboflow = Roboflow(api_key=api_key)
    project = roboflow.workspace(dataset_config["workspace"]).project(dataset_config["project"])
    version = project.version(version_id)
    downloaded = version.download("yolov11", location=str(destination), overwrite=True)

    dataset_root = locate_dataset_root(destination, getattr(downloaded, "location", None))
    source_yaml = dataset_root / "data.yaml"
    if not source_yaml.is_file():
        raise RuntimeError(f"Source data.yaml was not found at {source_yaml}.")

    source_config = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    names = normalize_names(source_config.get("names"))

    normalized_yaml = root / "data" / "dataset" / "data.yaml"
    normalized_yaml.parent.mkdir(parents=True, exist_ok=True)
    normalized_config = {
        "path": str(dataset_root),
        "train": str(dataset_root / "train" / "images"),
        "val": str(dataset_root / "valid" / "images"),
        "test": str(dataset_root / "test" / "images"),
        "names": names,
    }
    normalized_yaml.write_text(
        yaml.safe_dump(normalized_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(f"Dataset root verified: {dataset_root}")
    print(f"Normalized YOLO data YAML written: {normalized_yaml}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Dataset download failed: {error}", file=sys.stderr)
        raise SystemExit(1)
