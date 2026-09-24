# Kegagalan Eksperimen dan Pembersihan Repository

## Prinsip membaca kegagalan

`REJECTED` berarti kandidat tidak boleh dipakai sebagai model final. `UNRESOLVED` berarti penyebab teknis lengkapnya belum terbukti. File diagnostik tetap disimpan agar keputusan dapat diaudit; file model kandidat gagal yang besar dibersihkan. Penghapusan dari worktree dan commit baru tidak menghapus riwayat Git lama.

## 1. Export QAT lama melalui `nncf.torch.strip()`

Percobaan historis `restored QAT → strip → OpenVINO` menghasilkan graph yang terlihat terkuantisasi, tetapi catatan export menyebut relative MAE sekitar **36,24%** terhadap QAT PyTorch, jauh di atas batas penerimaan 1%. Graph quantized saja tidak membuktikan output ekuivalen. Angka historis ini berasal dari catatan skrip/fase lama; rincian per gambar untuk run strip tidak tersedia sebagai JSON lokal yang lengkap. Karena itu penyebab internal persis dari divergence strip tidak boleh diklaim pasti. Salinan IR lama memiliki SHA256 XML `0c1ad74c...e516aba2` dan BIN `8c21324a...c104c` dalam [metrik diagnostik VAL](../experiments/ptq/qat_rejected_openvino_val_metrics.json).

IR lama tersebut **bukan** `artifacts/openvino/qat_int8/` saat ini. Lokasi final sekarang berisi IR direct OpenVINO yang berbeda hash dan lolos audit. Salinan lama yang sempat di-stage di `experiments/ptq/backend_models/qat_openvino_model/` dihapus. Metrik VAL diagnostik tetap disimpan dengan label rejected; path `staged backend path` di JSON adalah jejak historis dan tidak lagi menunjuk file yang tersedia.

## 2. Jalur QAT → ONNX → OpenVINO

[Diagnostic ONNX](../artifacts/openvino/qat_onnx_export_diagnostic.json) menunjukkan export dan ONNX checker berhasil, tetapi kandidat OpenVINO gagal numerical equivalence pada **31 dari 32** citra VAL. Mean relative MAE **33,21%** dan maximum **37,55%**, melewati batas 1%. ONNX Runtime juga tidak dapat mengeksekusi representasi QDQ yang dicoba: error konkret melaporkan input `tensor(double)` tidak valid bagi operator `QuantizeLinear`. Field status diagnostic menyebut custom NNCF operator unsupported, sedangkan field error memberi alasan tipe data tersebut; kedua catatan tidak cukup untuk menentukan satu penyebab pasti. Bukti yang ada menunjukkan kandidat gagal pada jalur tersebut, tetapi tidak menetapkan mekanisme internal seluruh mismatch.

Kandidat `artifacts/openvino/debug/qat_onnx*` adalah intermediate yang ditolak dan tidak pernah dipromosikan. File ONNX/IR besar di empat direktori debug tersebut dihapus. JSON diagnostik ditahan sebagai bukti beserta status `REJECTED / UNRESOLVED`; path kandidat di dalamnya sekarang adalah path historis.

## 3. Synthetic CPU/CUDA QAT

Input sintetis lama pernah menunjukkan mismatch besar, sedangkan uji 32 citra VAL representatif berikutnya lulus relative MAE ≤1%. Ini menunjukkan kegagalan synthetic lama tidak cukup untuk menyimpulkan checkpoint QAT rusak. Detail dan batas interpretasinya ada di [diagnostik CPU/CUDA](11_cpu_cuda_qat_diagnostic.md). Tidak ada checkpoint yang dihapus karena percobaan ini.

## 4. Percobaan PTQ dan batch 128

Percobaan awal loader PTQ berhenti karena Ultralytics mengharapkan direktori `*_openvino_model`, bukan path `model.xml` langsung; [catatan PTQ](22_ptq_comparison.md) menjelaskan perbaikannya. Artifact PTQ canonical berikutnya lolos hash/provenance gate, sehingga tidak dihapus.

Run FP32 speed batch 128 tidak memiliki `results.csv` atau manifest hasil yang lengkap. Sebab penghentian terakhir tidak dipersistenkan, jadi tidak diklaim sebagai OOM yang pasti. Satu-satunya `args.yaml` run itu dihapus; run batch 16/32/64 yang memiliki hasil tetap disimpan sebagai diagnostik pemilihan konfigurasi.

## Apa yang dibersihkan

| Target | Alasan | Bukti yang masih tersedia |
| --- | --- | --- |
| `artifacts/openvino/debug/qat_onnx/`, `qat_onnx_openvino/`, `qat_onnx_qdq/`, `qat_onnx_qdq_openvino/` | Intermediate ONNX/IR rejected | `artifacts/openvino/qat_onnx_export_diagnostic.json` |
| `experiments/ptq/backend_models/qat_openvino_model/` | Salinan IR QAT lama rejected | `experiments/ptq/qat_rejected_openvino_val_metrics.json` |
| `experiments/debug/fp32_speed_b128_cache_false/` | Run tidak lengkap tanpa hasil | `docs/15_known_issues.md` dan riwayat Git |
| `artifacts/openvino/debug/direct_qat/` | Salinan lokal untracked identik SHA256 dengan QAT IR final | `artifacts/openvino/qat_int8/` |
| `scripts/export_openvino.py.bak` | Salinan lokal untracked identik SHA256 dengan skrip tracked | `scripts/export_openvino.py` |
| `experiments/ptq/qat_int8_rejected_val/` | Direktori kosong | Metrik diagnostik JSON tetap di `experiments/ptq/` |

Yang **tidak** dibersihkan: checkpoint FP32/QAT final, semua kandidat `qat_search` (kelima manifest kandidat berstatus `passed`; yang kalah tetap bukti seleksi VAL), PTQ comparator, metrik final TEST, JSON diagnostik, benchmark server, dan hasil hardware OP3 yang sudah ada. Kandidat lama tracked masih dapat dipulihkan dari commit sebelum pembersihan bila penelitian forensik memerlukannya; pemulihan bukan bagian dari alur deployment.

## Kenapa QAT final saat ini tetap valid

Percobaan gagal di atas mendahului promosi artifact QAT direct OpenVINO yang valid. [Diagnostic export final](../artifacts/openvino/qat_export_diagnostic.json) berstatus `accepted and promoted`, memilih `direct OpenVINO`, melaporkan 32/32 input VAL lulus batas relative MAE 1%, dan menunjukkan FakeQuantize serta elemen INT8 dalam graph. `qat_validation_gate()` memeriksa checkpoint, seleksi VAL, dan bukti export tersebut. Lihat [provenance](26_provenance_dan_verifikasi.md) untuk hash final.
