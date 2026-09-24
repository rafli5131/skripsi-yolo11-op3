# Known Issues

## Tujuan

Merekam masalah tanpa menyembunyikan statusnya.

## Status

**ACTIVE REFERENCE**.

| Isu | Gejala | Akar penyebab | Workaround | Status |
|---|---|---|---|---|
| Polygon Roboflow | satu label detection berbentuk polygon | export anomali; detail file tidak tersimpan | min/max polygon ke bbox, audit ulang | RESOLVED, provenance filename UNRESOLVED |
| Path absolute | debug args memuat `.../skripsi/...` lama | project berpindah | gunakan root `advance_vision` dan YAML canonical | KNOWN |
| NAS lambat | warning slow image access | filesystem mounted | cache false, workers 2; pindah lokal bila diizinkan | KNOWN |
| NNPACK | warning runtime | UNKNOWN | tidak dipakai sebagai hasil penelitian | UNKNOWN |
| Codex sandbox | CUDA tidak tersedia | sandbox berbeda dari host | validasi GPU hanya SSH host | EXPECTED |
| GPU mapping | physical 1 menjadi cuda:0 | CUDA_VISIBLE_DEVICES | gunakan device logis 0 | RESOLVED |
| Batch 128 benchmark | run tidak lengkap/interrupted | sebab final tidak dipersistenkan | batch 16 dipilih untuk shared server; `args.yaml` tanpa hasil dibersihkan | HISTORICAL |
| NNCF strip historis | relative MAE dilaporkan ~36,24% | output tidak ekuivalen; mekanisme internal belum dipastikan | jangan gunakan kandidat strip | REJECTED |
| Synthetic CPU/CUDA | 29,46% pada input lama | input-distribution-specific | gunakan `[0,1]` dan VAL representatif | RESOLVED for representative inputs |
| QAT IR lama | graph quantized tetapi tidak ekuivalen | jalur strip; mekanisme divergence internal belum pasti | salinan lama dibersihkan, gunakan IR final direct OpenVINO dengan hash/gate valid | HISTORICAL REJECTED |
| W&B index | UI dapat menunjukkan epoch 101 | convention/log indexing UNKNOWN | gunakan CSV/manifest untuk epoch aktual 100 | KNOWN |
| Jalur QAT -> ONNX -> OpenVINO lama | ONNX/checker lulus, kandidat OpenVINO gagal 31/32 input VAL | penyebab internal lengkap belum ditetapkan; ORT QDQ melaporkan tipe `tensor(double)` tidak valid untuk `QuantizeLinear` | jangan gunakan kandidat ini; direct OpenVINO final sudah accepted | HISTORICAL REJECTED |
| ONNX Runtime custom NNCF op | ONNX Runtime tidak dapat memuat `org.openvinotoolkit::FakeQuantize`. | Stock ORT tidak mendaftarkan custom operator fake-quant NNCF 2.13.0. | Catat sebagai intermediate execution unavailable; jangan strip/PTQ; lanjut hanya ke OpenVINO importer. | EXPECTED LIMITATION |
