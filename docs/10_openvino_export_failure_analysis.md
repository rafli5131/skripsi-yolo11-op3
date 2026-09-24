# Analisis Kegagalan Export QAT Lama

## Ruang lingkup

Dokumen ini membahas **kandidat historis yang ditolak**, bukan IR QAT final saat ini. IR final `artifacts/openvino/qat_int8/model.{xml,bin}` sekarang berasal dari direct OpenVINO dengan checkpoint SHA256 `b9e520...35430d` dan lolos gate. Baca [provenance](26_provenance_dan_verifikasi.md) untuk hash lengkap.

## Apa yang gagal

Jalur lama `restored NNCF QAT → nncf.torch.strip() → OpenVINO` menghasilkan output finite `[1,7,8400]` dan graph tampak terkuantisasi (112 FakeQuantize, 88 konstanta INT8, 97 Convert menurut catatan fase). Namun relative MAE sekitar 36,24% terhadap QAT PyTorch dilaporkan dalam catatan export, jauh di atas batas penerimaan 1%. Data per gambar untuk percobaan strip itu tidak dipersistenkan secara lengkap, sehingga angka tersebut adalah bukti historis terbatas. Penyebab internal persis divergence tidak terbukti hanya dari angka itu.

Kesalahan penilaian yang perlu dihindari: melihat operasi quantization pada graph lalu menyimpulkan model pasti ekuivalen. Gate harus memeriksa output numerik pada input VAL representatif dan identitas checkpoint sumber.

## Nasib artifact

Salinan IR strip lama dengan hash XML `0c1ad74c...e516aba2` dan BIN `8c21324a...c104c` pernah dipakai untuk [diagnostik VAL rejected](../experiments/ptq/qat_rejected_openvino_val_metrics.json). Salinan model yang di-stage di `experiments/ptq/backend_models/qat_openvino_model/` telah dihapus dalam pembersihan. JSON metrik dipertahankan agar status rejected dan hash historis tetap dapat diperiksa. Path di JSON tersebut bukan path model aktif lagi.

Percobaan ONNX setelahnya juga ditolak karena gagal numerical equivalence; baca [21 Percobaan ONNX](21_qat_onnx_openvino_export.md). Solusi final yang diterima adalah jalur direct OpenVINO yang berbeda, bukan menamai ulang artifact lama atau menjalankan PTQ baru. Rangkuman semua file yang dibersihkan ada di [25 Kegagalan dan pembersihan](25_kegagalan_dan_pembersihan.md).
