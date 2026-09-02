# QAT Smoke dan Validasi Integrasi

## Tujuan

Mencatat bukti kompatibilitas sebelum final QAT.

## Status

**DONE — DIAGNOSTIC ONLY**, bukan hasil penelitian akhir.

Phase 5A: 184 `SymmetricQuantizer`, 193 fake-quant calls, forward, loss deteksi YOLO, backward, dan satu `optimizer.step()` berhasil. Parameter trainable 2.599.029; 16 parameter DFL sengaja dibekukan.

Phase 5B: DetectionTrainer menjalankan 1 epoch (`fraction=0.1`, batch 8) dengan 26.332 fake-quant calls. Smoke validation: P 0,87795; R 0,92627; mAP50 0,95320; mAP50-95 0,64624. Reload memulihkan 1.323/1.323 entries dan 184 quantizer; forward dan validation berhasil, serta fingerprint bobot berbeda dari inisialisasi FP32.

## Artefak terkait

[`scripts/smoke_train_qat.py`](../scripts/smoke_train_qat.py), [manifest smoke](../experiments/debug/qat_smoke/manifest.json).
