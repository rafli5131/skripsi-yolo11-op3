"""Audit YOLO detection dataset integrity without modifying images or labels."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# This script has no secret-bearing settings, but consistently reads root .env.
load_dotenv(Path.cwd() / ".env", override=False)

import yaml


CLASS_NAMES = {0: "ball", 1: "gawang", 2: "robot"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitAudit:
    name: str
    images: int = 0
    labels: int = 0
    backgrounds: int = 0
    annotations: int = 0
    invalid: int = 0
    missing_image_label_pairs: int = 0
    orphan_labels: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)
    stems: set[str] = field(default_factory=set)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset/data.yaml"), help="Normalized YOLO YAML.")
    return parser.parse_args()


def image_files(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def label_files(labels_dir: Path) -> list[Path]:
    return sorted(path for path in labels_dir.rglob("*.txt") if path.is_file())


def validate_line(line: str) -> int | None:
    tokens = line.split()
    if len(tokens) != 5:
        return None
    try:
        class_id = int(tokens[0])
        values = [float(token) for token in tokens[1:]]
    except ValueError:
        return None
    if class_id not in CLASS_NAMES or not all(math.isfinite(value) for value in values):
        return None
    x_center, y_center, width, height = values
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return None
    return class_id


def audit_split(name: str, images_dir: Path) -> SplitAudit:
    labels_dir = images_dir.parent / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise RuntimeError(f"{name}: expected images and labels directories under {images_dir.parent}.")

    audit = SplitAudit(name=name)
    images = image_files(images_dir)
    labels = label_files(labels_dir)
    audit.images = len(images)
    audit.labels = len(labels)
    audit.stems = {image.stem for image in images}
    image_relatives = {image.relative_to(images_dir).with_suffix("") for image in images}

    for image in images:
        label_path = labels_dir / image.relative_to(images_dir).with_suffix(".txt")
        if not label_path.is_file():
            audit.backgrounds += 1
            audit.missing_image_label_pairs += 1
            continue

        nonempty_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not nonempty_lines:
            audit.backgrounds += 1
            continue
        for line in nonempty_lines:
            class_id = validate_line(line)
            if class_id is None:
                audit.invalid += 1
            else:
                audit.annotations += 1
                audit.class_counts[class_id] += 1

    for label in labels:
        if label.relative_to(labels_dir).with_suffix("") not in image_relatives:
            audit.orphan_labels += 1

    return audit


def print_split(audit: SplitAudit) -> None:
    print(f"\n{audit.name}")
    print(f"images: {audit.images}")
    print(f"labels: {audit.labels}")
    print(f"backgrounds: {audit.backgrounds}")
    print(f"annotations: {audit.annotations}")
    print(f"invalid: {audit.invalid}")
    print(f"missing image-label pairs: {audit.missing_image_label_pairs}")
    print(f"orphan labels: {audit.orphan_labels}")
    print("class distribution:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_id} {class_name}: {audit.class_counts[class_id]}")


def normalized_split_path(config: dict, key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Normalized YAML is missing '{key}'.")
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"Normalized YAML '{key}' path must be absolute, got: {value}")
    return path


def main() -> int:
    arguments = parse_arguments()
    yaml_path = arguments.data
    if not yaml_path.is_file():
        raise RuntimeError(f"Normalized dataset YAML not found: {yaml_path}")
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    audits = {
        "TRAIN": audit_split("TRAIN", normalized_split_path(config, "train")),
        "VAL": audit_split("VAL", normalized_split_path(config, "val")),
        "TEST": audit_split("TEST", normalized_split_path(config, "test")),
    }
    for audit in audits.values():
        print_split(audit)

    for left, right in (("TRAIN", "VAL"), ("TRAIN", "TEST"), ("VAL", "TEST")):
        overlap = audits[left].stems & audits[right].stems
        if overlap:
            sample = ", ".join(sorted(overlap)[:10])
            print(f"WARNING: filename stem overlap {left} vs {right}: {len(overlap)} (examples: {sample})", file=sys.stderr)

    invalid_total = sum(audit.invalid for audit in audits.values())
    empty_splits = [name for name, audit in audits.items() if audit.images == 0]
    if invalid_total or empty_splits:
        if invalid_total:
            print(f"Audit failed: {invalid_total} invalid annotation line(s).", file=sys.stderr)
        if empty_splits:
            print(f"Audit failed: split(s) with zero images: {', '.join(empty_splits)}.", file=sys.stderr)
        return 1

    print("\nAudit passed: all splits contain images and no invalid annotation lines were found.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Dataset audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
