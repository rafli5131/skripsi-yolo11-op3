# OpenVINO Export

## Tujuan

Mencatat export FP32 dan kriteria export QAT tanpa PTQ.

## Status

FP32 IR: **VALID**. QAT IR: **UNRESOLVED**; artefak `qat_int8` lama adalah **REJECTED**.

## FP32

Source adalah `artifacts/checkpoints/fp32/best.pt`; input NCHW `[1,3,640,640]`. Jalur yang dicatat manifest: `torch.jit.trace -> openvino.convert_model -> openvino.save_model(compress_to_fp16=False)`. IR berada di `artifacts/openvino/fp32/model.{xml,bin}` (XML 324.944 bytes, BIN 10.431.992 bytes). Manifest yang tersedia saat ini berstatus `started`, sehingga angka MAE 0,00594 dan magnitude 111,26 yang disebut pada catatan eksternal **tidak dapat diverifikasi dari manifest lokal**.

## QAT

Restore wajib: FP32 architecture -> `NNCFConfig` -> `create_compressed_model` -> compression state -> QAT state. Sebelum export: 1.323 entries dan 184 quantizer aktif. Candidate A adalah direct `ov.convert_model` tanpa `strip`; Candidate B adalah `compression_ctrl.export_model` hanya bila A gagal. Tidak ada `nncf.quantize()`, PTQ, calibration, atau TEST. Acceptance membutuhkan raw `[1,7,8400]`, representative relative MAE <=1%, dan graph FakeQuantize/I8/U8.

## Artefak terkait

[`scripts/export_openvino.py`](../scripts/export_openvino.py), [failure analysis](10_openvino_export_failure_analysis.md), [manifest](../artifacts/openvino/export_manifest.json).
