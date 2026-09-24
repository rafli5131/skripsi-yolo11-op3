# Ringkasan Hasil Training & Evaluasi YOLO11n (ROBOTIS OP3)

Ringkasan ringkas 1 halaman hasil training, kuantisasi (QAT vs PTQ), serta evaluasi deteksi dan benchmark kinerja model YOLO11n (deteksi `ball`, `gawang`, dan `robot`).

---

## 1. Perbandingan Kualitas Deteksi (VAL & Held-Out TEST)

Evaluasi kualitas deteksi pada dataset **VAL (1.213 gambar)** dan **TEST (592 gambar)**:

| Model / Metode | Precision | Recall | F1-Score | mAP50 | mAP50-95 | Evaluasi Set |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline (PyTorch)** | 0,9474 | 0,9465 | 0,9470 | 0,9734 | 0,7783 | VAL |
| **FP32 Baseline (PyTorch)** | 0,9313 | 0,9506 | 0,9409 | 0,9642 | 0,7799 | TEST |
| **QAT INT8 (Quantization-Aware)** | **0,9341** | **0,9502** | **0,9421** | **0,9721** | **0,7672** | VAL / TEST |
| **PTQ INT8 (Post-Training)** | 0,8657 | 0,9305 | 0,8970 | 0,9614 | 0,6549 | VAL |

> **Analisis Utama**: QAT INT8 berhasil menjaga performa deteksi mendekati FP32 (bahkan melampaui mAP50 FP32 pada TEST), sedangkan PTQ INT8 mengalami penurunan signifikan terutama pada mAP50-95 (turun dari ~0,778 ke 0,655).

---

## 2. Metrik Deteksi Per Kelas (QAT INT8 TEST Set)

Performa deteksi per kelas pada **QAT INT8** (592 gambar TEST):

| Objek | Precision | Recall | F1-Score | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ball (Bola)** | 0,9430 | 1,0000 | 0,9707 | 0,9892 | 0,8796 |
| **Gawang (Goalpost)** | 0,8988 | 0,8811 | 0,8899 | 0,9458 | 0,5970 |
| **Robot (Humanoid)** | 0,9606 | 0,9695 | 0,9650 | 0,9814 | 0,8250 |

---

## 3. Benchmark Kinerja & Efisiensi Server (OpenVINO CPU)

Pengukuran diagnostik benchmark pada server OpenVINO CPU (300 iterasi):

| Metrik Kinerja | FP32 OpenVINO | PTQ INT8 | QAT INT8 | Keunggulan QAT vs FP32 |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Latency (ms)** | 17,160 ms | 13,097 ms | **12,829 ms** | **-25,24% (-4,33 ms)** |
| **Inference FPS** | 58,28 FPS | 76,35 FPS | **77,95 FPS** | **+33,76% (+19,68 FPS)** |
| **P95 Latency (ms)** | 22,773 ms | 25,855 ms | **23,120 ms** | Lebih stabil dari PTQ |
| **Process CPU Mean** | 3231,92% | 2974,01% | **2865,06%** | **-11,35% penggunaan CPU** |
| **RAM Physical (RSS)** | 254,58 MiB | 238,01 MiB | **233,80 MiB** | **-20,78 MiB RAM** |
| **Ukuran File Model** | 10,26 MiB | 3,32 MiB | **3,25 MiB** | **-68,32% kompresi** |

---

## 4. Kesimpulan Ringkas

1. **Efisiensi Kuantisasi**: QAT INT8 menghasilkan kompresi model hingga **3,25 MiB (-68.32%)** dan meningkatkan FPS inference sebesar **+33.76%** (dari 58,28 FPS menjadi **77,95 FPS**).
2. **Kualitas Deteksi**: Model **QAT INT8** secara konsisten melampaui PTQ INT8 di seluruh metrik (F1: 0,9421 vs 0,8970; mAP50: 0,9721 vs 0,9614).
3. **Model Rekomendasi**: QAT INT8 merupakan model terbaik yang direkomendasikan untuk deployment di ROBOTIS OP3 karena unggul dalam akurasi, kecepatan latency, serta efisiensi resource CPU/RAM.
