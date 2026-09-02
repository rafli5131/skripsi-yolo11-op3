# Ringkasan Hasil Penelitian

## Tujuan

Merangkum metrik validasi dan held-out TEST tanpa menarik kesimpulan deployment.

## Status

**FP32/QAT TEST DONE; deployment benchmark NOT YET EXECUTED**.

| Ruang evaluasi | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| QAT best validation (epoch 1) | 0,8810 | 0,9393 | 0,9585 | 0,6513 |
| QAT final validation (epoch 5) | 0,8748 | 0,9448 | 0,9610 | 0,6510 |
| FP32 TEST | 0,9313 | 0,9506 | 0,9642 | 0,7799 |
| QAT TEST | 0,8542 | 0,9375 | 0,9512 | 0,6567 |

TEST delta QAT terhadap FP32: P -0,0772 (-8,29%), R -0,0131 (-1,37%), mAP50 -0,0130 (-1,35%), mAP50-95 -0,1232 (-15,80%), F1 -0,0470 (-4,99%). Per-class mAP50-95: ball 0,9130/0,7470; gawang 0,5683/0,5282; robot 0,8585/0,6948 (FP32/QAT).

Tidak ada kesimpulan latency, FPS, CPU, RAM, atau manfaat deployment karena benchmark OpenVINO CPU dan deployment OP3 belum dilakukan.

## Artefak terkait

[QAT final](07_final_qat_training.md), [TEST](08_final_test_evaluation.md), [comparison JSON](../experiments/final_test/comparison.json).
