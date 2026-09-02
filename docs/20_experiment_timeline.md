# Timeline Eksperimen

## Tujuan

Memberikan kronologi input, aksi, hasil, dan artefak.

## Status

**CURRENT THROUGH PHASE 7D PREPARATION**.

| Phase | Goal/aksi | Hasil | Artefak/status |
|---|---|---|---|
| 1--2 | environment dan project setup | uv/config siap | DONE |
| 3 | download/audit dataset | split dan audit passed | DONE |
| 3.5 | benchmark batch | batch 16 dipilih; b128 issue | DIAGNOSTIC |
| 4 | FP32 100 epoch | baseline final | DONE |
| 5A | probe API QAT | 184 quantizer, step berhasil | DONE |
| 5B | smoke trainer/reload | 1 epoch dan restore valid | DONE |
| 5C | QAT final 5 epoch | best epoch 1 | DONE |
| 6 | held-out TEST | FP32/QAT comparison frozen | DONE |
| 7A | FP32/QAT OpenVINO | FP32 valid; strip QAT rejected | MIXED |
| 7B | export diagnosis | synthetic old CPU/CUDA mismatch | STOPPED |
| 7C | representative CPU/CUDA | `[0,1]` + 32 VAL pass | DONE |
| 7D | direct QAT export preparation | script/docs prepared | NOT YET EXECUTED |
| 7F preparation | QAT -> ONNX -> OpenVINO | controller NNCF resmi, checker ONNX, dan gate numerik/graph disiapkan | NOT YET EXECUTED |
