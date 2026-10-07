# OpenVINO Export

## Status model saat ini

FP32 IR valid. PTQ comparator existing lolos provenance gate. QAT INT8 final sekarang memakai **direct OpenVINO** dan lolos numerical equivalence/provenance gate. Jalur strip dan ONNX sebelumnya hanya merupakan eksperimen pengembangan dan tidak digunakan sebagai file model aktif.

| Model | Artifact | Sumber |
| --- | --- | --- |
| FP32 | `artifacts/openvino/fp32/model.{xml,bin}` | `artifacts/checkpoints/fp32/best.pt` |
| PTQ INT8 | `experiments/ptq/backend_models/ptq_openvino_model/model.{xml,bin}` | FP32 melalui PTQ TRAIN historis |
| QAT INT8 | `artifacts/openvino/qat_int8/model.{xml,bin}` | `artifacts/checkpoints/qat/qat_best.pt` melalui direct OpenVINO |

## FP32

[Export manifest](../artifacts/openvino/export_manifest.json) mencatat `torch.jit.trace → openvino.convert_model → openvino.save_model(compress_to_fp16=False)` dengan input `[1,3,640,640]`. Model FP32 menjadi baseline dan sumber PTQ. File IR final dicatat pada manifest dan inventaris artefak.md).

## QAT final

Checkpoint QAT final adalah true NNCF QAT dengan 184 quantizer yang dipulihkan. [Diagnostic export final](../artifacts/openvino/qat_export_diagnostic.json) memilih `direct OpenVINO` dan berstatus `accepted and promoted`: 32/32 citra VAL memenuhi relative MAE ≤1%, graph memiliki 112 FakeQuantize dan 88 elemen INT8/U8, `PTQ used=false`, `calibration occurred=false`, dan TEST tidak dimuat. `qat_validation_gate()` menggabungkan bukti checkpoint, seleksi VAL, dan export itu.

## Riwayat kandidat yang tidak diterima

Beberapa jalur export awal pernah diuji sebagai bagian eksplorasi implementasi. Artefak final yang digunakan sekarang adalah direct OpenVINO yang telah tervalidasi pada input VAL representatif. Catatan eksperimen lama dipertahankan hanya sebagai riwayat pengembangan dan tidak mewakili status QAT final.

Export QAT final yang valid tidak memerlukan PTQ fallback, calibration baru, retraining, atau TEST. Benchmark server ketiga IR tersedia di [laporan server](23_server_openvino_benchmark.md); benchmark final OP3 masih terpisah.
