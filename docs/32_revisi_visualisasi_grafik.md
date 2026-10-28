# Revisi skala grafik hasil penelitian

Revisi ini memperjelas selisih kecil dari data eksperimen yang sudah selesai. Angka sumber,
checkpoint, pembagian data, dan protokol pengujian tidak diubah. Tidak ada training atau
pengukuran robot baru. Grafik yang dihasilkan tersedia di [galeri grafik](figures/README.md).

## Grafik yang perlu diperjelas

| Kelompok | Masalah pada tampilan sebelumnya | Perubahan |
| --- | --- | --- |
| Lima kandidat QAT | mAP50–95 sekitar 0,65–0,66 tampak hampir sama pada batang dari nol | Konteks batang dari nol disandingkan dengan titik berskala lokal dan label tiga desimal persen |
| Kualitas TEST keseluruhan | Precision, recall, dan F1 berbeda kurang dari satu poin persentase | Grafik selisih bertanda dan panel titik dengan empat desimal persen |
| Kualitas per kelas | Rata-rata keseluruhan menyembunyikan kelas yang membaik atau menurun | Perbandingan tiap kelas untuk seluruh lima metrik, dengan konteks dan detail |
| Riwayat metrik FP32 | Perubahan akhir training sulit dibaca setelah kenaikan awal yang besar | Kurva seluruh 100 epoch dan grafik terpisah 20 epoch terakhir |
| Metrik setiap kandidat QAT | Satu rentang 0–1 meratakan fluktuasi kecil di tiap metrik | Panel per metrik dengan sumbu lokal yang diberi keterangan |
| RSS dan RAM ROS | Selisih kecil terhadap besar total, ditambah perbedaan kondisi diam/gerak | Batang dari nol dan titik detail pada skala yang sama untuk seluruh perbandingan |
| Resource runtime | CPU sistem dan proses, atau RAM sistem dan RSS, berbagi satu sumbu padahal besarannya jauh berbeda | Grafik time series dipisahkan menurut cakupan pengukuran |
| Standalone empat model | Detail PTQ–QAT sulit terbaca bersama nilai baseline yang jauh lebih besar | Konteks dan panel detail dari angka yang sama |

### Contoh selisih yang sekarang terlihat

- C3 vs C4: mAP50–95 VAL **0,66340 vs 0,66472**, atau **+0,132 poin persentase**.
- FP32 vs QAT pada TEST: mAP50 **+0,7914 poin persentase**, sedangkan mAP50–95
  **−1,2730 poin persentase**. Precision **+0,2769**, recall **−0,0364**, F1 **+0,1232** poin.
- RSS detector saat diam: FP32 **271,4608 MB**, QAT **257,6271 MB**.
  Saat gerak didampingi: FP32 **271,7696 MB**, QAT **257,1154 MB**.

Pembesaran memperjelas data, bukan memperbesar manfaat numeriknya. Kandidat C4 memang
tertinggi pada VAL, tetapi jarak terhadap C3 tetap kecil. QAT tidak unggul pada setiap metrik.
FPS ROS final lebih rendah pada QAT walaupun mean inferensinya lebih cepat. Seluruh arah
perubahan tetap ditampilkan. Replikasi training dan interval ketidakpastian tidak tersedia,
sehingga grafik tidak menambahkan error bar atau klaim signifikansi yang tidak dihitung.

## Aturan sumbu dan satuan

1. Batang untuk nilai absolut selalu berawal dari nol. Panjang batang tetap proporsional.
2. Perbandingan detail memakai **titik** dengan rentang sumbu yang tertulis, bukan batang
   yang dipotong. Konteks dan detail memakai angka identik.
3. Grafik selisih memiliki garis nol dan mempertahankan tanda positif/negatif. Selisih skor
   deteksi dinyatakan dalam **poin persentase**, bukan persentase perubahan relatif.
4. Sumbu lokal pada time series dan panel metrik dinyatakan pada keterangan. Seluruh titik
   dalam cakupan waktu ditampilkan. Grafik 20 epoch terakhir melengkapi grafik penuh.
5. CPU proses multi-core dapat melebihi 100%. RSS proses berbeda dari RAM sistem maupun
   ukuran berkas model. Nilai MB memakai satu juta byte.
6. Perbandingan lintas konteks tetap terpisah: VAL untuk seleksi, TEST untuk kualitas
   checkpoint PyTorch, server diagnostik, standalone NUC12, dan ROS final.

Pemilihan bentuk mengikuti [panduan bar chart ONS](https://service-manual.ons.gov.uk/data-visualisation/chart-types/bar-charts)
dan [sumbu/grid ONS](https://service-manual.ons.gov.uk/data-visualisation/guidance/axes-and-gridlines):
jika nol menutupi perbedaan, gunakan bentuk berbasis posisi seperti dot plot.

## Membuat ulang grafik

Jalankan dari root repository dengan lingkungan Python yang memiliki Matplotlib:

```bash
python -m pip install -r scripts/reporting/requirements.txt
python scripts/reporting/generate_thesis_figures.py
python -m unittest scripts.reporting.test_figure_style -v
```

Hasil PNG berada di `docs/figures/`. Untuk keluaran vektor:

```bash
python scripts/reporting/generate_thesis_figures.py --output /tmp/thesis-figures --formats png svg pdf
```

Skrip tidak mengimpor PyTorch, Ultralytics, NNCF, atau OpenVINO dan tidak memerlukan GPU,
model Git LFS, kamera, maupun dataset mentah. Jalankan ke folder baru jika ingin menyimpan
versi gambar sebelumnya. Hanya keluaran grafik yang ditulis ulang, bukan CSV/JSON eksperimen.

`figure_inventory.json` mencatat sumber setiap gambar, catatan interpretasi, dan batas sumbu
untuk perbandingan detail. Gambar keluaran evaluator di `experiments/` tetap menjadi arsip;
PR curve dan confusion matrix tidak direkonstruksi dari skor agregat.

## Pemakaian pada skripsi dan PPT

- Grafik kandidat terbaik: `qat_candidate_best.png` dan `qat_candidate_history.png`.
- Training FP32: `fp32_loss.png`, `fp32_metrics.png`, `fp32_metrics_last20.png`.
- Kandidat QAT: `qat_c1_*` sampai `qat_c5_*`.
- TEST: `test_overall_and_delta.png`, `test_overall_focused.png`, `test_class_*.png`.
- Standalone NUC12: `standalone_synthetic_*` dan `standalone_camera_*`.
- ROS final: `ros_*.png`. Stabilitas: `stability_*.png`.

Pilih grafik detail ketika membahas selisih kecil. Konteks besaran tetap ada pada gambar
pasangan. Dokumen Word dan PPT yang sudah diekspor tidak otomatis berubah oleh pembaruan
repository ini; gambar baru dapat menggantikan gambar lama dengan judul/caption yang sesuai.
