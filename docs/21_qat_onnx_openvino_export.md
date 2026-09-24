# Percobaan QAT → ONNX → OpenVINO (Historis)

## Tujuan saat percobaan

Jalur ini mencoba memisahkan export QAT PyTorch menjadi dua tahap, ONNX lalu OpenVINO, untuk mengetahui di mana numerical mismatch muncul. Skrip penelitian adalah [`scripts/export_qat_onnx_openvino.py`](../scripts/export_qat_onnx_openvino.py); bukti run terdapat pada [diagnostic JSON](../artifacts/openvino/qat_onnx_export_diagnostic.json). Percobaan ini **sudah dijalankan dan ditolak**. Ia bukan jalur model QAT final saat ini.

## Hasil yang tercatat

- Checkpoint QAT historis yang dipakai memiliki SHA256 `f96911a1b3f4f9a8987a7fb57611aac2455b2b5c7b93c27db146d69d5013a8c8`; ini berbeda dari checkpoint final terpilih kemudian (`b9e520...35430d`). Perbandingan antarjalur tidak boleh mengabaikan perbedaan checkpoint ini.
- Export ONNX dan `onnx.checker` lulus. ONNX graph memakai pasangan `QuantizeLinear`/`DequantizeLinear`.
- ONNX Runtime tidak dapat mengeksekusi kandidat QDQ itu. Error konkret: input `tensor(double)` tidak valid untuk `QuantizeLinear`. Field status diagnostic menyebut custom NNCF operator unsupported, tetapi tidak cocok sebagai penjelasan langsung untuk error QDQ tersebut. Penyebab spesifik eksekusi ONNX Runtime perlu dipisahkan dari penyebab mismatch output OpenVINO.
- Konversi ONNX → OpenVINO berjalan, tetapi output OpenVINO gagal uji numerik: **1/32 VAL lulus**, **31/32 gagal** pada batas relative MAE 1%; mean relative MAE 33,21% dan maksimum 37,55%.
- Status diagnostik adalah `REJECTED / UNRESOLVED`, promotion `not promoted`, `PTQ used=false`, `calibration occurred=false`, dan TEST tidak dimuat. Masalah komputasi internal spesifik yang membuat output berbeda belum ditetapkan secara definitif.

## Keputusan

Kandidat ONNX/IR di `artifacts/openvino/debug/qat_onnx*` tidak digunakan untuk benchmark atau deployment. File model intermediate telah dibersihkan; path kandidat di JSON adalah catatan historis. Artifact QAT final yang ada sekarang memakai **direct OpenVINO** dari checkpoint final terpilih dan telah lolos gate terpisah. Baca [export saat ini](09_openvino_export.md), [pembersihan](25_kegagalan_dan_pembersihan.md), dan [provenance](26_provenance_dan_verifikasi.md).
