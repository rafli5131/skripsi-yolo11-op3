# Provenance dan Verifikasi Artefak

## Tujuan

Verifikasi artefak memastikan bahwa model yang digunakan sesuai dengan jalur penelitian, class mapping, checkpoint terpilih, dan metode kuantisasi yang benar. Pemeriksaan tidak bergantung pada pencocokan nilai identitas berkas.

## Artefak canonical

| Model | XML | BIN | Dasar verifikasi |
| --- | --- | --- | --- |
| FP32 | `artifacts/openvino/fp32/model.xml` | `artifacts/openvino/fp32/model.bin` | baseline FP32 final dan pasangan IR tersedia |
| PTQ INT8 | `experiments/ptq/backend_models/ptq_openvino_model/model.xml` | `experiments/ptq/backend_models/ptq_openvino_model/model.bin` | PTQ-only manifest, source FP32, class mapping, dan pasangan IR tersedia |
| QAT INT8 | `artifacts/openvino/qat_int8/model.xml` | `artifacts/openvino/qat_int8/model.bin` | true NNCF QAT, winner VAL, direct OpenVINO, dan numerical equivalence |

## Gate

Untuk gate spesifik PTQ/QAT tanpa training, gunakan fungsi `ptq_validation_gate()` dan `qat_validation_gate()` di `scripts/op3_model_audit.py`. Gate yang siap digunakan harus menghasilkan:

```text
accepted: true
reasons: []
```

Gate QAT memeriksa provenance true QAT, checkpoint terpilih, isolasi TEST/PTQ, numerical equivalence, quantizer, dan graph OpenVINO. Gate PTQ memeriksa provenance PTQ-only, class mapping, serta keberadaan pasangan XML/BIN.

## Status akhir

QAT final menggunakan direct OpenVINO dan berstatus **VALID**. Artefak final inilah yang digunakan untuk benchmark serta deployment ROBOTIS OP3.
