# OpenVINO Export

## Status model saat ini

FP32 IR valid. PTQ comparator existing lolos hash/provenance gate. QAT INT8 final sekarang memakai **direct OpenVINO** dan lolos numerical equivalence/provenance gate. Kandidat strip dan ONNX lama yang ditolak adalah artifact berbeda dan tidak lagi ada sebagai file model aktif di worktree.

| Model | Artifact | Sumber |
| --- | --- | --- |
| FP32 | `artifacts/openvino/fp32/model.{xml,bin}` | `artifacts/checkpoints/fp32/best.pt` |
| PTQ INT8 | `experiments/ptq/backend_models/ptq_openvino_model/model.{xml,bin}` | FP32 melalui PTQ TRAIN historis |
| QAT INT8 | `artifacts/openvino/qat_int8/model.{xml,bin}` | `artifacts/checkpoints/qat/qat_best.pt` melalui direct OpenVINO |

## FP32

[Export manifest](../artifacts/openvino/export_manifest.json) mencatat `torch.jit.trace → openvino.convert_model → openvino.save_model(compress_to_fp16=False)` dengan input `[1,3,640,640]`. Model FP32 menjadi baseline dan sumber PTQ. File IR final diidentifikasi oleh SHA256 dalam [provenance](26_provenance_dan_verifikasi.md).

## QAT final

Checkpoint QAT final SHA256 `b9e520cbd5d56597014c932701f471505d2243e02763a2245a51fd83fc35430d` adalah true NNCF QAT dengan 184 quantizer yang dipulihkan. [Diagnostic export final](../artifacts/openvino/qat_export_diagnostic.json) memilih `direct OpenVINO` dan berstatus `accepted and promoted`: 32/32 citra VAL memenuhi relative MAE ≤1%, graph memiliki 112 FakeQuantize dan 88 elemen INT8/U8, `PTQ used=false`, `calibration occurred=false`, dan TEST tidak dimuat. `qat_validation_gate()` menggabungkan bukti checkpoint, seleksi VAL, dan export itu.

## Riwayat kandidat yang tidak diterima

Jalur strip lama gagal numerical equivalence; jalur ONNX/QDQ berikutnya juga gagal pada 31/32 input VAL. Keduanya tidak boleh dipakai sebagai QAT final. File model kandidat besar dibersihkan, tetapi JSON diagnostik dan analisisnya masih ada: [kegagalan strip](10_openvino_export_failure_analysis.md), [percobaan ONNX](21_qat_onnx_openvino_export.md), dan [inventaris pembersihan](25_kegagalan_dan_pembersihan.md).

Export QAT final yang valid tidak memerlukan PTQ fallback, calibration baru, retraining, atau TEST. Benchmark server ketiga IR tersedia di [laporan server](23_server_openvino_benchmark.md); benchmark final OP3 masih terpisah.
