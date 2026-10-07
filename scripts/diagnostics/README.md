# Diagnostic Tools

Folder ini berisi verifikasi teknis yang mendukung pipeline utama tetapi
**bukan** langkah metodologi utama BAB III.

- `probe_qat_compatibility.py`: kompatibilitas true NNCF QAT dengan YOLO11n.
- `smoke_train_qat.py`: lifecycle trainer, checkpoint, reload, dan quantizer.
- `diagnose_qat_cpu_cuda.py`: diagnosis ekuivalensi backend.
- `diagnose_qat_export_state.py`: diagnosis state export.
- `diagnose_system.py`, `diagnose_gpu.py`, `verify_torch_cuda.py`: audit environment.

Pipeline utama berada di `scripts/step01_*.py` sampai
`scripts/step07_*.py`.
