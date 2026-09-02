# Analisis Kegagalan Export QAT

## Tujuan

Menetapkan alasan artefak QAT lama tidak boleh dideploy.

## Status

**REJECTED / DO NOT DEPLOY**.

Jalur `restored NNCF QAT -> nncf.torch.strip() -> OpenVINO` menghasilkan divergence sekitar 36,24% terhadap restored trained QAT menurut komentar historis pada script export, melebihi batas 1%. Kedua output dilaporkan finite dan berbentuk `[1,7,8400]`; graph lama berisi 112 FakeQuantize, 88 konstanta int8, dan 97 Convert menurut catatan fase sebelumnya. Mean absolute difference 41,59 tidak dipersistenkan dalam JSON yang tersisa, sehingga **UNVERIFIED from local artifact**.

Graph yang tampak terkuantisasi tidak membuktikan equivalence semantik. Oleh karena itu `artifacts/openvino/qat_int8/model.xml` dan `.bin` dipertahankan hanya sebagai artefak **REJECTED**, tidak untuk benchmark/deployment. Tidak ada fallback `nncf.quantize()`, PTQ, kalibrasi Ultralytics, atau retuning.

## Artefak terkait

[export script](../scripts/export_openvino.py), [diagnostic lama](../artifacts/openvino/qat_export_diagnostic.json), [Phase 7C](11_cpu_cuda_qat_diagnostic.md).
