"""STEP 07 — Seleksi checkpoint QAT terbaik berbasis VAL.

Entry point berurutan untuk kebutuhan laporan skripsi.
Implementasi utama tetap berada di scripts/tune_qat_best.py agar kode eksperimen
yang sudah tervalidasi tidak diduplikasi atau diubah.

Jalankan dari root repository:
    uv run python scripts/report_steps/07_select_qat_best.py [argumen]
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
TARGET = SCRIPTS_DIR / "tune_qat_best.py"


def main() -> None:
    """Jalankan implementasi canonical untuk STEP 07."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
