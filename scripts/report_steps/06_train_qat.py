"""STEP 06 — Training final QAT.

Entry point berurutan untuk kebutuhan laporan skripsi.
Implementasi utama tetap berada di scripts/train_qat.py agar kode eksperimen
yang sudah tervalidasi tidak diduplikasi atau diubah.

Jalankan dari root repository:
    uv run python scripts/report_steps/06_train_qat.py [argumen]
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
TARGET = SCRIPTS_DIR / "train_qat.py"


def main() -> None:
    """Jalankan implementasi canonical untuk STEP 06."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
