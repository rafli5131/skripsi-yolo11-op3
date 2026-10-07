"""STEP 03 — Training baseline YOLO11n FP32.

Entry point berurutan untuk kebutuhan laporan skripsi.
Implementasi utama tetap berada di scripts/train_fp32.py agar kode eksperimen
yang sudah tervalidasi tidak diduplikasi atau diubah.

Jalankan dari root repository:
    uv run python scripts/report_steps/03_train_fp32.py [argumen]
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
TARGET = SCRIPTS_DIR / "train_fp32.py"


def main() -> None:
    """Jalankan implementasi canonical untuk STEP 03."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
