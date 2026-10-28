# Pemilihan grafik akademis: dasar riset dan audit repositori

Ditinjau 28 Oktober 2026. Tujuannya memilih bentuk sesuai pertanyaan, cakupan pengukuran,
dan jumlah observasi, bukan membuat seluruh hasil tampak meningkat. Tidak ada jenis grafik
yang menjamin persetujuan penguji; pilihan berikut memiliki alasan yang dapat dijelaskan.

## Apa yang dipakai penelitian lain?

1. **Jacob et al., Quantization and Training of Neural Networks for Efficient
   Integer-Arithmetic-Only Inference.** Versi arXiv v1, bagian 4.2 dan Gambar 4.1–4.2,
   menyajikan hubungan latency–accuracy pada perangkat tertentu. Tabel 4.4 menyandingkan
   kualitas deteksi dan waktu. Contohnya menunjukkan bahwa perbandingan harus menjaga
   konteks perangkat dan evaluasi, bukan bahwa semua hasil harus berupa batang.
   [Naskah lengkap](https://arxiv.org/html/1712.05877v1).
2. **Chen et al., AQD: Towards Accurate Fully-Quantized Object Detection.** Versi
   arXiv v5, Tabel 2, membandingkan AP, AP50, AP75, dan ukuran objek menurut model/presisi.
   Tabel cocok untuk banyak metrik yang harus dibaca tepat. Contoh ini mendukung tetap
   menyediakan tabel angka bersama grafik ringkas, bukan menghubungkan semua metrik
   menjadi satu kurva. [PDF lengkap](https://arxiv.org/pdf/2007.06919).
3. **Nature Biomedical Engineering, Show the dots in plots (2017).** Editorial
   menyarankan menampilkan observasi individual, terutama sampel kecil. Ini mendasari
   tampilan tiga titik run, bukan menyembunyikan pengulangannya di balik rata-rata.
   [Artikel](https://doi.org/10.1038/s41551-017-0079). Halaman penuh mengalami pengalihan
   akses saat penelusuran; rekomendasi ini dibatasi pada ringkasan penerbit yang tersedia.
4. **ONS, Choosing a chart type dan Bar chart.** Pemilihan mengikuti hubungan yang
   hendak dijelaskan. Connected-dot dapat memperlihatkan besaran dan selisih; batang
   memakai nol sebagai dasar panjang. [Pemilihan bentuk](https://service-manual.ons.gov.uk/data-visualisation/chart-types/choosing-a-chart-type),
   [aturan batang](https://service-manual.ons.gov.uk/data-visualisation/chart-types/bar-charts).
5. **CDC, Box-and-Whiskers Plot.** Box plot digunakan untuk distribusi, dengan median,
   kuartil, dan aturan outlier yang dijelaskan. Karena repositori menyimpan latensi mentah,
   distribusinya dapat ditampilkan, bukan hanya mean.
   [Panduan](https://www.cdc.gov/cove/data-visualization-types/box-and-whiskers-plot.html).

Penerapan di bawah adalah keputusan untuk data penelitian ini, bukan ketentuan wajib
yang diklaim berasal dari semua paper tersebut. Nomor gambar/tabel merujuk versi yang ditautkan.

## Data seperti apa memakai grafik apa?

| Data dan pertanyaan | Bentuk utama | Alasan dan batasnya |
| --- | --- | --- |
| Jumlah citra/anotasi tiap split | Batang mulai nol, tabel angka | Membandingkan jumlah diskret. Citra dan anotasi tidak ditumpuk karena bukan bagian dari total yang sama. |
| Lima kandidat QAT, skor terbaik | Dot plot berlabel dan tabel | Kandidat merupakan kategori, bukan waktu; tidak perlu garis penghubung C1–C5. Skala lokal membuat selisih kecil terbaca. |
| Loss dan metrik menurut epoch | Line chart, panel per metrik | Epoch benar-benar berurutan. TRAIN/VAL berbagi skala dalam panel; panel berbeda tidak dibaca sebagai besar perubahan yang setara. |
| FP32–QAT pada TEST atau per kelas | Connected-dot/dumbbell, grafik delta, tabel | Penghubung membandingkan dua skor agregat dalam kategori yang sama. Bukan pasangan citra atau bukti signifikansi. |
| Lima jenis metrik TEST | Small multiples | Precision, recall dan mAP tidak disambungkan sebagai deret waktu. Selisih persen dinyatakan dalam poin persentase. |
| Mean/P95/FPS/CPU/RSS model | Batang konteks nol + titik detail | Membaca besaran dan selisih. Mean/P95 tetap ringkasan; tidak menggantikan distribusi mentah. |
| Tiga run ROS per model/kondisi | Strip/dot plot seluruh run + mean | n=3 terlihat langsung. Tidak ada garis lintas run/model yang menyiratkan tren atau pasangan pengukuran. Mean di sini antar-run, bukan pooled-frame mean. |
| Ratusan latensi dalam satu sesi | Box plot dan ECDF | Box menunjukkan median, Q1–Q3, whisker 1,5 IQR, dan outlier. ECDF menunjukkan proporsi observasi di bawah ambang tanpa pemilihan bin. Bukan ratusan replikasi independen. |
| Latensi/CPU/RSS terhadap detik | Time series, panel terpisah | Menampilkan perubahan dan lonjakan sepanjang pengamatan. Ringkasan 5 detik melengkapi, tidak menghapus observasi asli. |
| Precision–recall | PR curve evaluator | Memerlukan pasangan precision/recall pada berbagai ambang; tidak dapat dibuat dari satu skor mAP. |
| Salah prediksi antar kelas | Confusion-matrix heatmap | Perlu hitungan asli dan definisi normalisasi. Matriks arsip tidak dilabel ulang sebagai evaluasi final yang berbeda. |
| Accuracy–latency trade-off | Scatter jika evaluasi dan artefak sebanding | Belum dibuat untuk TEST-versus-ROS di sini: TEST checkpoint PyTorch dan runtime IR tidak membentuk pasangan evaluasi berlabel deployment yang terverifikasi. |

## Hasil audit dan perubahan yang dilakukan

- Bentuk yang sudah sesuai dipertahankan: kurva epoch, time series stabilitas, grafik delta,
  konteks batang nol, dan panel titik. Semua hasil naik maupun turun tetap tampil.
- Panel detail FP32–QAT memakai **dumbbell** dengan titik pada kategori yang sejajar.
  Penghubung hanya menunjukkan selisih agregat dalam kategori itu.
- Dua grafik keterulangan ROS diganti dari garis antar-run menjadi **tiga titik per model
  dan penanda mean antar-run**. Nomor run yang sama bukan bukti pasangan lintas model.
- Grafik distribusi server menjelaskan **whisker 1,5 IQR** dan menampilkan outlier;
  **ECDF** ditambahkan untuk membaca proporsi latensi tanpa asumsi bentuk distribusi.
- Seluruh gambar hasil generator diindeks dalam `chart_selection_audit.csv`, dengan
  keluarga grafik, sumber, keputusan, dan batas interpretasi. Galeri:
  [docs/figures](figures/README.md).
- Grafik evaluator dan gambar eksperimen historis tidak ditimpa. Bentuk PR curve,
  confusion matrix, dan bukti deteksi sesuai fungsinya; perbedaan arsip QAT terhadap
  JSON final tetap perlu disebut saat digunakan, sebagaimana revisi sebelumnya.

## Pilihan yang sengaja dihindari

Tidak memakai radar untuk membandingkan metrik yang perbedaannya kecil, 3D, dua sumbu
untuk CPU dan memori, batang dengan dasar terpotong, atau box plot untuk tiga skor
agregat. Tidak membuat error bar/CI dari satu skor TEST. SD/SEM/CI tidak dipertukarkan.
Kalau kelak ada replikasi training, ketidakpastian kualitas dapat dihitung sesuai unit
replikasinya. Distribusi frame satu sesi tidak menggantikan replikasi eksperimen.

## Membuat ulang dan memeriksa

```bash
python scripts/reporting/generate_thesis_figures.py
python -m unittest scripts.reporting.test_figure_style -v
```

Generator menulis PNG dan audit setiap gambar. Opsi `--formats png svg pdf` menghasilkan
format vektor. Sebelum dimasukkan ke dokumen, pilih grafik yang menjawab pembahasan;
tabel detail tetap disediakan untuk angka tepat. Word/PPT yang sudah diekspor memerlukan
penggantian gambar tersendiri. Pedoman format khusus FILKOM dan arahan pembimbing tetap
menjadi acuan tata letak akhir.
