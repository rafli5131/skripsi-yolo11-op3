# Catatan Export QAT

## Status saat ini

Model QAT final **berhasil** digunakan dan artefak OpenVINO final berada di `artifacts/openvino/qat_int8/model.{xml,bin}`. Artefak tersebut berasal dari jalur direct OpenVINO dan telah melewati validasi numerik menggunakan input VAL representatif.

## Catatan eksperimen awal

Selama pengembangan pernah dicoba beberapa jalur export alternatif, termasuk transformasi melalui `strip()` dan ONNX. Jalur-jalur tersebut hanya digunakan untuk eksplorasi implementasi dan **bukan** artefak QAT final yang digunakan untuk deployment.

Hasil akhir penelitian menggunakan checkpoint QAT terpilih, quantizer NNCF yang telah dipulihkan, dan artefak direct OpenVINO yang tervalidasi. Dengan demikian, status yang berlaku untuk model QAT adalah valid dan siap digunakan pada tahap deployment.

Lihat [export OpenVINO](09_openvino_export.md), [seleksi QAT](23_qat_best_selection.md), dan [provenance](26_provenance_dan_verifikasi.md) untuk status final.
