# Decision Log

## Tujuan

Merekam keputusan kronologis, alasan, bukti, dan dampaknya.

## Status

**CURRENT**.

| Phase | Keputusan | Alasan/bukti | Dampak |
|---|---|---|---|
| Setup | YOLO11n dan dataset detection native | config dan class mapping | baseline konsisten |
| Dataset | pipeline segmentation ditinggalkan; satu polygon dikoreksi | audit passed | label valid, provenance file terbatas |
| Server | uv, batch 16, workers 2, cache false | shared-server safety | run lebih konservatif |
| FP32 | 100 epoch AdamW | manifest FP32 | baseline dibekukan |
| 5A | tolak `nncf.quantize`, pilih `create_compressed_model` | docstring/API NNCF | true QAT |
| 5B | integrate DetectionTrainer dan NNCF-aware checkpoint | smoke/reload passed | QAT lifecycle valid |
| 5C | LR 1e-5, no warmup, lrf 1.0 | 5 epoch dari FP32 converged | QAT final terkunci |
| 6 | model dibekukan sebelum TEST; TEST sekali | manifest final test | tidak ada test tuning |
| 7A | tolak strip artifact | divergence ~36,24% | no deploy artifact |
| 7C/7D | acceptance memakai `[0,1]` + 32 VAL | CPU/CUDA representative pass | export QAT fail-closed |
| 7F preparation | Memakai `QuantizationController.export_model(..., save_format="onnx")` sebagai jalur utama | Controller NNCF 2.13.0 terpasang menyediakan API ekspor ONNX resmi untuk compressed model. | Jalur ONNX dipisahkan dari diagnostik OpenVINO langsung; tidak ada fallback PTQ. |
