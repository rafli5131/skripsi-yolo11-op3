# Evaluasi Deteksi Server pada VAL — Empat Model

## Pertanyaan dan batas eksperimen

Pengujian ini membandingkan kualitas deteksi empat artifact yang sudah dibekukan pada **split VAL** yang sama: FP32 native PyTorch, FP32 OpenVINO, PTQ INT8 OpenVINO, dan QAT INT8 OpenVINO. VAL berisi 1.213 gambar dan 1.525 anotasi. Evaluator Ultralytics 8.4.130 berjalan pada CPU dengan `imgsz=640`, batch 1, workers 0, fraction 1.0, augmentasi off, `conf=None` (default), dan IoU 0.7. Model PTQ/QAT lolos provenance gate sebelum dievaluasi; setiap source file dicocokkan dengan SHA256 final.

Ini evaluasi kualitas deteksi di server, **bukan FINAL TEST** dan bukan benchmark latency final ROBOTIS OP3. Tidak ada training, export, quantization, atau calibration baru. Skrip dan raw result ada di [`scripts/evaluate_server_val.py`](../scripts/evaluate_server_val.py) dan [direktori eksperimen](../experiments/server_openvino/README.md).

## Hasil keseluruhan

| Metrik | FP32 native | FP32 OpenVINO | PTQ INT8 | QAT INT8 |
| --- | ---: | ---: | ---: | ---: |
| Precision | 0.9474 | 0.9362 | 0.9341 | 0.8657 |
| Recall | 0.9465 | 0.9576 | 0.9502 | 0.9305 |
| F1 | 0.9470 | 0.9468 | 0.9421 | 0.8970 |
| mAP50 | 0.9734 | 0.9730 | 0.9721 | 0.9614 |
| mAP50-95 | 0.7783 | 0.7713 | 0.7672 | 0.6549 |

F1 di sini adalah harmonic mean dari precision dan recall keseluruhan yang dikembalikan evaluator. Nilai mentah tanpa pembulatan ada di [comparison CSV](../experiments/server_openvino/val_comparison.csv) dan [JSON](../experiments/server_openvino/val_comparison.json); [laporan eksperimen](../experiments/server_openvino/VAL_METRICS_REPORT.md) memuat tabel per kelas.

## Membaca perbedaannya

FP32 OpenVINO hampir sama dengan FP32 native pada F1 (−0,02 poin persentase), tetapi precision turun 1,12 poin dan recall naik 1,11 poin; mAP50-95 turun 0,69 poin. Konversi format dapat menggeser keseimbangan precision/recall meskipun F1 total hampir sama.

PTQ dibanding FP32 OpenVINO turun 0,47 poin F1 dan 0,42 poin mAP50-95 pada VAL ini. PTQ tetap dekat dengan baseline dalam mAP50 (selisih sekitar 0,08 poin). Ini perbandingan kualitas; [benchmark sintetis server](23_server_openvino_benchmark.md) secara terpisah menunjukkan PTQ lebih cepat dan lebih kecil pada host ini.

QAT menghasilkan F1 0,8970 dan mAP50-95 0,6549, lebih rendah daripada PTQ masing-masing sekitar 4,51 dan 11,22 poin. Penurunan mAP50-95 terlihat pada ketiga kelas, terutama `ball` (0,7119 vs PTQ 0,8796) dan `robot` (0,6971 vs 0,8250). QAT memiliki precision `gawang` 0,9195 yang lebih tinggi daripada PTQ 0,8988, tetapi recall `gawang` lebih rendah (0,8389 vs 0,8811). Jadi kesimpulan per metrik dan kelas tidak selalu sama.

QAT OpenVINO yang dievaluasi tetap artifact **valid**: gate menyatakan export direct OpenVINO ekuivalen secara numerik dengan sumber QAT-nya pada input VAL audit. Gate itu tidak menyatakan QAT harus mengungguli FP32 atau PTQ dalam accuracy. QAT berasal dari proses training dan checkpoint berbeda dari FP32; selisih kualitas deteksi VAL ini dilaporkan apa adanya.

## Memisahkan hasil

Benchmark latency sebelumnya memakai satu frame sintetis dan 300 iterasi per IR; evaluasi ini memakai VAL berlabel. Waktu inference yang dicatat validator dalam JSON hanya diagnostik evaluator dan tidak menggantikan benchmark latency terkontrol. [Hasil FINAL TEST PyTorch](08_final_test_evaluation.md) berasal dari split TEST dan konteks lain; angkanya tidak dicampur ke tabel VAL ini. Perbandingan deployment final tetap memerlukan ketiga IR diuji langsung pada ROBOTIS OP3.
