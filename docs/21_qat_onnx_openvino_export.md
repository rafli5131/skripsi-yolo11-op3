# Catatan Jalur QAT → ONNX → OpenVINO

## Tujuan

Jalur ONNX pernah dieksplorasi sebagai alternatif untuk membawa model QAT menuju OpenVINO. Eksperimen ini membantu memisahkan tahap QAT PyTorch, representasi ONNX, dan konversi OpenVINO.

## Status saat ini

Artefak QAT final yang digunakan untuk penelitian dan deployment **bukan** bergantung pada jalur ONNX tersebut. Model final menggunakan **direct OpenVINO** dari checkpoint QAT terpilih dan telah melewati validasi numerik pada 32 citra VAL representatif.

## Kesimpulan

Jalur ONNX tetap dicatat sebagai eksplorasi implementasi, tetapi tidak digunakan sebagai sumber artefak final. Status model QAT final adalah **VALID** dan siap dilanjutkan ke deployment.

Lihat [export OpenVINO](09_openvino_export.md), [seleksi QAT](23_qat_best_selection.md), dan [provenance](26_provenance_dan_verifikasi.md).
