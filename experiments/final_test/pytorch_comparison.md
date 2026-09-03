# Perbandingan FP32 dan QAT PyTorch

## Status

**DONE — laporan dibuat dari hasil evaluasi native PyTorch yang sudah dibekukan.**
Tidak ada model yang dimuat dan tidak ada inference/evaluasi ulang yang dijalankan.

Split: `TEST (frozen existing reports; no re-evaluation)`; images: `592`; annotations: `734`.

## Metrik keseluruhan

| Metrik | FP32 PyTorch | QAT PyTorch | Delta absolut | Delta relatif |
|---|---:|---:|---:|---:|
| precision | 0.931347 | 0.854151 | -0.077196 | -8.288660% |
| recall | 0.950572 | 0.937520 | -0.013051 | -1.373002% |
| F1 | 0.940861 | 0.893896 | -0.046965 | -4.991730% |
| mAP50 | 0.964232 | 0.951187 | -0.013045 | -1.352900% |
| mAP50-95 | 0.779908 | 0.656703 | -0.123205 | -15.797403% |
| Latency (ms/image; diagnostic GPU) | 2.352011 | 16.262865 | 13.910854 | 591.445038% |
| FPS (derived; diagnostic GPU) | 425.168032 | 61.489780 | -363.678252 | -85.537535% |

## Metrik per kelas

| Kelas | Metrik | FP32 PyTorch | QAT PyTorch | Delta absolut | Delta relatif |
|---|---|---:|---:|---:|---:|
| ball | precision | 0.972209 | 0.919899 | -0.052309 | -5.380458% |
| ball | recall | 0.987952 | 0.975904 | -0.012048 | -1.219512% |
| ball | F1 | 0.980017 | 0.947074 | -0.032943 | -3.361445% |
| ball | mAP50 | 0.984762 | 0.983573 | -0.001189 | -0.120700% |
| ball | mAP50-95 | 0.912989 | 0.747049 | -0.165940 | -18.175494% |
| gawang | precision | 0.846951 | 0.865082 | 0.018131 | 2.140703% |
| gawang | recall | 0.885338 | 0.867779 | -0.017560 | -1.983378% |
| gawang | F1 | 0.865720 | 0.866428 | 0.000709 | 0.081871% |
| gawang | mAP50 | 0.923352 | 0.913192 | -0.010160 | -1.100366% |
| gawang | mAP50-95 | 0.568273 | 0.528224 | -0.040049 | -7.047546% |
| robot | precision | 0.974881 | 0.777471 | -0.197410 | -20.249654% |
| robot | recall | 0.978425 | 0.968879 | -0.009546 | -0.975680% |
| robot | F1 | 0.976650 | 0.862685 | -0.113965 | -11.668924% |
| robot | mAP50 | 0.984581 | 0.956795 | -0.027786 | -2.822156% |
| robot | mAP50-95 | 0.858462 | 0.694836 | -0.163626 | -19.060374% |

## Interpretasi

- Precision, recall, mAP, dan F1 berasal dari evaluasi native PyTorch pada TEST yang telah dilakukan sebelumnya.
- Latency/FPS adalah pengukuran diagnostik GPU dari evaluator, bukan benchmark deployment ROBOTIS OP3.
- CPU utilization dan RAM tidak tersedia pada laporan ini; keduanya memerlukan benchmark terkontrol di perangkat target.
- Perbedaan ini membandingkan kualitas model PyTorch FP32 dengan model PyTorch QAT, bukan validitas ekspor OpenVINO.
- TEST tidak dibaca ulang, tidak dipakai untuk training, kalibrasi, tuning, atau pemilihan checkpoint dalam pembuatan laporan ini.

## Artefak terkait

- FP32 report: `/mnt/nas-hpg9/rafli5131/rafli/advance_vision/experiments/final_test/fp32_test_metrics.json`
- QAT report: `/mnt/nas-hpg9/rafli5131/rafli/advance_vision/experiments/final_test/qat_test_metrics.json`
- JSON comparison: `/mnt/nas-hpg9/rafli5131/rafli/advance_vision/experiments/final_test/pytorch_comparison.json`
- CSV comparison: `/mnt/nas-hpg9/rafli5131/rafli/advance_vision/experiments/final_test/pytorch_comparison.csv`
