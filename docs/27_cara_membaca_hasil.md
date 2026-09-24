# Cara Membaca Hasil Eksperimen

## Empat konteks angka

| Konteks | Lokasi | Input/split | Pertanyaan yang dijawab |
| --- | --- | --- | --- |
| Seleksi checkpoint | `artifacts/checkpoints/qat/selection_manifest.json` | TRAIN untuk belajar, VAL untuk memilih | Kandidat QAT mana yang dipromosikan? |
| Final TEST | `experiments/final_test/` | Held-out TEST, evaluator PyTorch | Seberapa baik deteksi FP32 dan QAT final pada split TEST? |
| Evaluasi PTQ diagnostik | `experiments/ptq/` | VAL, OpenVINO | Apakah PTQ comparator mempertahankan metrik VAL terhadap FP32? |
| Benchmark server | `experiments/server_openvino/` | Satu frame sintetis 640×640, 300 iterasi | Berapa latency/FPS/resource ketiga IR pada host server? |
| Benchmark hardware OP3 | `experiments/hardware_op3/` | Sesuai laporan hardware setempat | Apa yang sudah diukur pada perangkat/lingkungan hardware tersebut? |

Jangan menggabungkan mAP dari TEST dengan latency dari benchmark server menjadi satu hasil eksperimen. Model, split, evaluator, dan hardware harus disebut setiap kali mengutip angka. Benchmark server terbaru dibandingkan dalam [laporan FP32/PTQ/QAT](23_server_openvino_benchmark.md). Hasil server tidak menggantikan benchmark final langsung di ROBOTIS OP3.

## Metrik deteksi

Precision adalah bagian prediksi positif yang benar; recall adalah bagian objek benar yang terdeteksi. mAP50 memakai ambang IoU 0,50, sedangkan mAP50-95 merata-ratakan beberapa ambang dari 0,50 hingga 0,95. F1 menyeimbangkan precision dan recall. Nilai ini hanya dapat dibandingkan secara adil bila dataset split dan konfigurasi evaluator sebanding.

## Metrik inference

Mean latency adalah rata-rata waktu inference; median/P50 adalah nilai tengah. P95 berarti 95% iterasi selesai pada atau di bawah nilai itu, sedangkan P99 menyoroti ekor distribusi. FPS inference di laporan server dihitung sebagai `1000 / mean latency ms`. Effective end-to-end FPS juga mencakup preprocess dan postprocess, sehingga tidak identik dengan FPS inference.

Process CPU dari `psutil` dapat lebih dari 100% karena 100% mewakili satu logical CPU. RSS adalah memori fisik proses yang menetap; VMS adalah ruang alamat virtual, sehingga VMS tidak boleh dibaca sebagai RAM fisik yang dipakai. Ukuran XML + BIN adalah ukuran file model di disk, bukan RSS saat runtime.

## Menarik kesimpulan

Satu run berurutan cukup untuk sanity check, tetapi tidak membuktikan signifikansi statistik. Periksa mean dan p95/p99 sekaligus, lalu perhatikan CPU, RSS, dan ukuran model. Kandidat dengan FPS tertinggi belum tentu memakai CPU/RAM paling rendah. Untuk klaim deployment OP3, ulangi pengukuran pada OP3 dengan input, konfigurasi, dan kondisi beban yang terkontrol.
