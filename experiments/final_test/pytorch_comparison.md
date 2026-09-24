# Perbandingan FP32 dan QAT PyTorch

## Status

**DONE — laporan dibuat dari hasil evaluasi native PyTorch yang sudah dibekukan.**
Tidak ada model yang dimuat dan tidak ada inference/evaluasi ulang yang dijalankan.

Split: `TEST (frozen existing reports; no re-evaluation)`; images: `592`; annotations: `734`.

## Metrik keseluruhan

| Metrik | FP32 PyTorch | QAT PyTorch | Delta absolut | Delta relatif |
|---|---:|---:|---:|---:|
| precision | 0.931347 | 0.934116 | +0.002769 | +0.297283% |
| recall | 0.950572 | 0.950208 | -0.000364 | -0.038266% |
| F1 | 0.940861 | 0.942093 | +0.001232 | +0.130941% |
| mAP50 | 0.964232 | 0.972145 | +0.007914 | +0.820706% |
| mAP50-95 | 0.779908 | 0.767179 | -0.012730 | -1.632185% |
| Latency (ms/image; diagnostic GPU) | 2.352011 | 16.262865 | 13.910854 | 591.445038% |
| FPS (derived; diagnostic GPU) | 425.168032 | 61.489780 | -363.678252 | -85.537535% |

## Metrik per kelas

| Kelas | Metrik | FP32 PyTorch | QAT PyTorch | Delta absolut | Delta relatif |
|---|---|---:|---:|---:|---:|
| ball | precision | 0.972209 | 0.942999 | -0.029210 | -3.004472% |
| ball | recall | 0.987952 | 1.000000 | +0.012048 | +1.219512% |
| ball | F1 | 0.980017 | 0.970663 | -0.009354 | -0.954439% |
| ball | mAP50 | 0.984762 | 0.989191 | +0.004429 | +0.449787% |
| ball | mAP50-95 | 0.912989 | 0.879553 | -0.033435 | -3.662191% |
| gawang | precision | 0.846951 | 0.898776 | +0.051825 | +6.118983% |
| gawang | recall | 0.885338 | 0.881132 | -0.004206 | -0.475085% |
| gawang | F1 | 0.865720 | 0.889867 | +0.024147 | +2.789266% |
| gawang | mAP50 | 0.923352 | 0.945807 | +0.022455 | +2.431896% |
| gawang | mAP50-95 | 0.568273 | 0.597027 | +0.028754 | +5.059812% |
| robot | precision | 0.974881 | 0.960572 | -0.014309 | -1.467755% |
| robot | recall | 0.978425 | 0.969492 | -0.008933 | -0.913032% |
| robot | F1 | 0.976650 | 0.965011 | -0.011638 | -1.191675% |
| robot | mAP50 | 0.984581 | 0.981437 | -0.003144 | -0.319300% |
| robot | mAP50-95 | 0.858462 | 0.824956 | -0.033507 | -3.903117% |

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
