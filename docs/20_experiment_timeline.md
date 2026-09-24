# Timeline Eksperimen

## Tujuan

Memberikan kronologi input, aksi, hasil, dan artefak.

## Status

**Kronologi hingga validasi QAT direct OpenVINO dan benchmark diagnostik server.**

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
| 7D/7F historis | QAT -> ONNX -> OpenVINO | ONNX/checker lulus, OpenVINO gagal 31/32 VAL; rejected | REJECTED |
| Seleksi QAT best | pencarian TRAIN/VAL, promosi winner | `qat_lr1e4_r256_e15`, TEST tidak untuk memilih | DONE |
| Direct OpenVINO final | export QAT terpilih dan audit 32 VAL | equivalence/graph/provenance accepted | VALID |
| Benchmark server | FP32 vs PTQ vs QAT, OpenVINO CPU | 30 warmup dan 300 timed iterations per model | DIAGNOSTIC DONE |
| Deployment OP3 final | benchmark synthetic dan kamera pada robot | perlu pengukuran langsung | NEXT |
