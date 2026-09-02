# Inventaris Artefak

## Tujuan

Mencatat artefak yang ditemukan tanpa membaca checkpoint biner.

## Status

**CURRENT INVENTORY**.

| Artefak | Lokasi | Tujuan/status | SHA256 bila tersedia |
|---|---|---|---|
| FP32 best/last | `artifacts/checkpoints/fp32/` | baseline final / DONE | best `e05b...20b8c` |
| QAT best/last | `artifacts/checkpoints/qat/` | final QAT / DONE | best `f969...3a8c8` |
| NNCF config/format | `artifacts/checkpoints/qat/*.json` | restore QAT / VALID | tercatat di manifest |
| FP32 IR | `artifacts/openvino/fp32/{model.xml,model.bin}` | OpenVINO FP32 / VALID | **SHA256 UNRESOLVED** |
| QAT IR | `artifacts/openvino/qat_int8/{model.xml,model.bin}` | **REJECTED / DO NOT DEPLOY** | **UNRESOLVED** |
| Final TEST metrics | `experiments/final_test/` | evaluasi final / DONE | n/a |
| FP32/QAT manifests | `experiments/fp32`, `experiments/qat` | provenance run / DONE | n/a |
| QAT diagnostics | `artifacts/openvino/*diagnostic.json` | debugging / lihat status masing-masing | n/a |

`experiments/` menyimpan run debug, FP32, QAT, dan final test; `artifacts/` menyimpan checkpoint freeze serta OpenVINO. Jalur absolute lama `.../skripsi/...` pada beberapa debug args adalah issue historis, bukan project root aktif.

## Artefak terkait

[kegagalan export](10_openvino_export_failure_analysis.md), [hasil](17_research_results_summary.md).
