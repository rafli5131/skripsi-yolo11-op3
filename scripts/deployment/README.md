# Deployment Tools

Folder ini khusus untuk implementasi dan pengukuran pada ROBOTIS OP3.

- `audit_hardware_op3.py`: audit perangkat dan runtime robot.
- `audit_op3_models.py`: audit artefak model untuk deployment.
- `benchmark_op3.py`: benchmark OpenVINO standalone.
- `benchmark_op3_pytorch.py`: benchmark PyTorch pembanding.
- `benchmark_op3_ros2.py`: benchmark pipeline ROS 2.
- `capture_op3_evidence.py`: capture bukti visual/telemetry.

Pipeline penelitian utama tetap berada pada `scripts/step01_*.py` sampai
`scripts/step07_*.py`. Tool di folder ini dipakai setelah STEP 07.
# Perbandingan runtime empat model

Jalankan dari root repository pada robot diam dengan kamera audited terpasang:

```bash
bash scripts/run_op3_model_comparison.sh
```

Runner memakai checkpoint `artifacts/checkpoints/fp32/best.pt` untuk PyTorch CPU tanpa OpenVINO, serta IR `artifacts/openvino/{fp32,ptq_int8,qat_int8}/model.xml` untuk OpenVINO CPU. Keempat model diukur berurutan: warmup 30, synthetic 300 iteration, kamera 60 detik, batch 1, input 640×640, visualization OFF. `rect=False` memastikan baseline PyTorch juga memakai input persegi 640×640 pada frame kamera 640×480.

Output baru berada di `experiments/hardware_op3/comparison_models/<timestamp>/`. `comparison_all.csv` berisi semua model; `comparison_thesis_fp32_qat.csv` mengecualikan PTQ untuk tabel skripsi. Accuracy checkpoint native PyTorch tetap dipisahkan dari runtime dan kamera. Data lama tidak ditimpa. Kamera live berurutan tidak menjamin scene/frame identik; implementasi preprocessing/postprocess dan timing stage masing-masing runtime berbeda.

## Menu operator satu script

Jalankan `bash scripts/run_op3.sh` untuk memilih FP32 OpenVINO / QAT INT8 / PTQ INT8 / FP32 PyTorch tanpa OpenVINO, lalu pilih visual, benchmark, atau kontrol gerak explicit. Panduan: [docs/30_operator_script.md](../../docs/30_operator_script.md). Memilih model tidak memulai walking. Mode ROS tersedia bagi OpenVINO; native PyTorch tersedia pada standalone dan visual.
