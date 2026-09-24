# Evaluasi Held-out TEST Final

## Tujuan

Mendokumentasikan satu-satunya evaluasi TEST final untuk model yang telah dikunci.

## Status

**DONE — TEST FROZEN**.

TEST berisi 592 images dan 734 annotations. Evaluator Ultralytics native PyTorch memakai imgsz 640, batch 16, logical cuda:0, workers 2, cache false, augment false, IoU 0.7; confidence default `conf=None`/0.001 identik dan tidak dituning.

| Metrik | FP32 | QAT | Delta absolut | Delta relatif |
|---|---:|---:|---:|---:|
| Precision | 0,9313 | 0,9341 | +0,0028 | +0,30% |
| Recall | 0,9506 | 0,9502 | -0,0004 | -0,04% |
| mAP50 | 0,9642 | 0,9721 | +0,0079 | +0,82% |
| mAP50-95 | 0,7799 | 0,7672 | -0,0127 | -1,63% |
| F1 | 0,9409 | 0,9421 | +0,0012 | +0,13% |

Per-class mAP50-95: ball 0,9130 -> 0,8796; gawang 0,5683 -> 0,5970; robot 0,8585 -> 0,8250. TEST tidak dipakai untuk training, range init, tuning, maupun checkpoint selection.

## Artefak terkait

[`scripts/evaluate_final_test.py`](../scripts/evaluate_final_test.py), [manifest](../experiments/final_test/manifest.json), [comparison](../experiments/final_test/comparison.json).
