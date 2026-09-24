# Pelatihan QAT Final

> Catatan status: dokumen ini merekam run QAT 5 epoch yang lebih awal. Checkpoint QAT final yang berlaku sekarang adalah pemenang pencarian VAL `qat_lr1e4_r256_e15` dengan SHA256 `b9e520cbd5d56597014c932701f471505d2243e02763a2245a51fd83fc35430d`; lihat [seleksi QAT best](23_qat_best_selection.md) dan [provenance](26_provenance_dan_verifikasi.md). Angka di bawah adalah riwayat run lama.

## Tujuan

Mencatat konfigurasi QAT final yang dikunci.

## Status

**RUN HISTORIS SELESAI; BUKAN CHECKPOINT FINAL TERKINI**.

Konfigurasi: epochs 5, fraction 1.0, batch 16, workers 2, imgsz 640, seed 42, AdamW, `lr0=1e-5`, momentum 0.9, weight decay 0.0005, AMP false, cache false, validation true. `warmup_epochs=0.0` menggantikan default 3.0 dan `lrf=1.0` menggantikan default 0.01: fine-tuning singkat dimulai dari FP32 yang sudah konvergen sehingga LR konstan 1e-5 tanpa warmup.

Range init: 16 TRAIN images deterministik, batch 2. Best epoch 1: P 0,88100; R 0,93929; mAP50 0,95852; mAP50-95 0,65134. Final epoch mAP50 0,96101 dan mAP50-95 0,65096. Run historis ini pernah memiliki checkpoint, manifest, config struktur NNCF, YAML, dan results CSV sendiri. Setelah pencarian QAT best, checkpoint canonical dipromosikan dari kandidat pemenang yang berbeda.

## Artefak terkait

[`scripts/train_qat.py`](../scripts/train_qat.py), [manifest run historis](../experiments/qat/qat_final/manifest.json), [CSV run historis](../experiments/qat/qat_final/results.csv), [manifest checkpoint final terkini](../artifacts/checkpoints/qat/manifest.json).
