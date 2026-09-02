# Dataset

## Tujuan

Menjelaskan struktur, validasi, dan isolasi split dataset Roboflow.

## Status

**AUDIT PASSED** menurut catatan eksperimen; distribusi kelas per split dan nama file anomali tidak tersimpan dalam artefak yang tersedia.

## Struktur dan ukuran

Workspace Roboflow `humanoid-atom`, project `advance_vision_human`, tipe Object Detection; versi Roboflow **UNRESOLVED** (`null` di manifest). Dataset canonical adalah [`data/dataset/data.yaml`](../data/dataset/data.yaml), yang menunjuk ke `data/raw/advance_vision_human/{train,valid,test}/images`.

| Split | Images | Anotasi | Background |
|---|---:|---:|---:|
| Train | 4.600 | 6.143 | 4 |
| Val | 1.213 | 1.525 | 2 |
| Test | 592 | 734 | 1 |

Class mapping: ball, gawang, robot. Tabel jumlah anotasi per kelas adalah **UNRESOLVED** karena output audit tidak dipersistenkan; script audit dapat melaporkannya, tetapi dokumentasi ini tidak menjalankannya.

## Anomali dan audit

Satu label TRAIN pernah diekspor sebagai polygon segmentation walau dataset detection. Perbaikannya adalah polygon ke bounding box YOLO dengan min/max koordinat x/y. Nama file dan class anomali **UNRESOLVED** dalam source/manifes saat ini, sehingga tidak diklaim. Audit memeriksa pasangan image/label, label kosong/background, orphan label, class ID, finite value, dan validitas `xc,yc,w,h`; hasil akhir dicatat passed. TEST tidak disentuh sampai evaluasi final Phase 6.

## Artefak terkait

[`scripts/download_dataset.py`](../scripts/download_dataset.py), [`scripts/audit_dataset.py`](../scripts/audit_dataset.py), [evaluasi TEST](08_final_test_evaluation.md).
