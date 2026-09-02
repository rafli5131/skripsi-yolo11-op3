# Pelatihan QAT Final

## Tujuan

Mencatat konfigurasi QAT final yang dikunci.

## Status

**DONE**.

Konfigurasi: epochs 5, fraction 1.0, batch 16, workers 2, imgsz 640, seed 42, AdamW, `lr0=1e-5`, momentum 0.9, weight decay 0.0005, AMP false, cache false, validation true. `warmup_epochs=0.0` menggantikan default 3.0 dan `lrf=1.0` menggantikan default 0.01: fine-tuning singkat dimulai dari FP32 yang sudah konvergen sehingga LR konstan 1e-5 tanpa warmup.

Range init: 16 TRAIN images deterministik, batch 2. Best epoch 1: P 0,88100; R 0,93929; mAP50 0,95852; mAP50-95 0,65134. Final epoch mAP50 0,96101 dan mAP50-95 0,65096. Artifact freeze menyimpan `qat_best.pt`, `qat_last.pt`, manifest, config struktur NNCF, format checkpoint, YAML, serta results CSV. Reload `qat_best.pt` lulus.

## Artefak terkait

[`scripts/train_qat.py`](../scripts/train_qat.py), [manifest](../artifacts/checkpoints/qat/manifest.json), [CSV](../artifacts/checkpoints/qat/results.csv).
