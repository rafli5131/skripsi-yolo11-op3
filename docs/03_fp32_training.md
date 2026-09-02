# Pelatihan FP32

## Tujuan

Mendokumentasikan baseline YOLO11n FP32 yang telah dibekukan.

## Status

**DONE**.

## Konfigurasi

Pretrained source `yolo11n.pt`; YOLO11n, epochs 100, patience 20, batch 16, workers 2, imgsz 640, device logis 0, seed 42, AdamW, `lr0=0.001111`, momentum 0.9, weight decay 0.0005, AMP true, cache false. W&B diaktifkan pada run final menurut manifest. Training selesai pada 2026-09-02; CSV merekam epoch 1 sampai 100 dan waktu kumulatif epoch 100 sebesar 3385,53 s. GPU memory puncak tidak dipersistenkan di manifest.

Epoch terakhir memiliki precision 0,94746, recall 0,94695, mAP50 0,97380, mAP50-95 0,77852. Ini bukan otomatis metrik `best.pt`; checkpoint terpilih adalah `best.pt`, sedangkan `last.pt` adalah keadaan epoch terakhir.

## Artefak terkait

[`scripts/train_fp32.py`](../scripts/train_fp32.py), [manifest](../artifacts/checkpoints/fp32/manifest.json), [CSV](../artifacts/checkpoints/fp32/results.csv), `artifacts/checkpoints/fp32/{best,last}.pt`.
