# Diagnostic Tools

Folder ini berisi verifikasi teknis yang mendukung pipeline utama tetapi
**bukan** langkah metodologi utama BAB III.

Yang paling penting:

- `probe_qat_compatibility.py`: memastikan true NNCF QAT kompatibel dengan YOLO11n.
- `smoke_train_qat.py`: memastikan lifecycle trainer, checkpoint, reload, dan quantizer berjalan.
- `diagnose_qat_cpu_cuda.py`: diagnosis ekuivalensi backend.
- `diagnose_qat_export_state.py`: diagnosis state export.
- `diagnose_system.py`, `diagnose_gpu.py`, dan `verify_torch_cuda.py`: audit environment.

Pipeline utama tetap berada di `scripts/step01_*.py` sampai
`scripts/step07_*.py`.
