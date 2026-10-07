# Catatan Eksperimen dan Pembersihan Repository

## Tujuan

Dokumen ini mencatat beberapa jalur implementasi yang pernah dicoba selama pengembangan serta pembersihan artefak debug yang tidak dipakai lagi. Catatan tersebut tidak mengubah status model QAT final.

## Status QAT final

Checkpoint QAT terpilih berasal dari true NNCF QAT. Artefak deployment final menggunakan **direct OpenVINO** dan telah melewati validasi numerik pada 32 citra VAL representatif. Graph OpenVINO juga mempertahankan struktur kuantisasi yang dibutuhkan.

Dengan demikian, status QAT final adalah **VALID** dan artefak tersebut yang digunakan untuk benchmark serta deployment.

## Eksperimen export awal

Selama pengembangan pernah dicoba jalur `strip()`, ONNX, serta beberapa variasi wrapper export. Jalur-jalur tersebut berfungsi sebagai eksperimen implementasi untuk menemukan alur export yang paling konsisten. Artefak intermediate dari eksperimen tersebut tidak digunakan sebagai model final.

## Pembersihan

File model debug berukuran besar yang tidak lagi diperlukan telah dipisahkan atau dibersihkan agar tidak tertukar dengan artefak final. JSON diagnostic dapat tetap dipertahankan sebagai catatan proses pengembangan.

Artefak yang menjadi rujukan deployment:

- FP32 OpenVINO: `artifacts/openvino/fp32/model.{xml,bin}`
- QAT INT8 OpenVINO: `artifacts/openvino/qat_int8/model.{xml,bin}`
- PTQ comparator: `experiments/ptq/backend_models/ptq_openvino_model/model.{xml,bin}`

Lihat [export OpenVINO](09_openvino_export.md), [inventaris artefak](14_artifact_inventory.md), dan [provenance](26_provenance_dan_verifikasi.md).
