# Evaluasi Held-out TEST Final

## Tujuan

Mendokumentasikan satu-satunya evaluasi TEST final untuk model yang telah dikunci.

## Status

**DONE — TEST FROZEN**.

TEST berisi 592 images dan 734 annotations. Evaluator Ultralytics native PyTorch memakai imgsz 640, batch 16, logical cuda:0, workers 2, cache false, augment false, IoU 0.7; confidence default `conf=None`/0.001 identik dan tidak dituning.

| Metrik | FP32 | QAT | Delta absolut | Delta relatif |
|---|---:|---:|---:|---:|
| Precision | 0,9313 | 0,8542 | -0,0772 | -8,29% |
| Recall | 0,9506 | 0,9375 | -0,0131 | -1,37% |
| mAP50 | 0,9642 | 0,9512 | -0,0130 | -1,35% |
| mAP50-95 | 0,7799 | 0,6567 | -0,1232 | -15,80% |
| F1 | 0,9409 | 0,8939 | -0,0470 | -4,99% |

Per-class mAP50-95: ball 0,9130 -> 0,7470; gawang 0,5683 -> 0,5282; robot 0,8585 -> 0,6948. mAP50 relatif dipertahankan, sedangkan kualitas lokalisasi lebih ketat menurun; penyebab kausal tidak disimpulkan dari hasil ini. TEST tidak dipakai untuk training, range init, tuning, maupun checkpoint selection.

## Artefak terkait

[`scripts/evaluate_final_test.py`](../scripts/evaluate_final_test.py), [manifest](../experiments/final_test/manifest.json), [comparison](../experiments/final_test/comparison.json).
