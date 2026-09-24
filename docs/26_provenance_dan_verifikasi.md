# Provenance Model dan Cara Verifikasi

## Tiga artifact yang berlaku

| Model | XML | BIN | Dasar penerimaan |
| --- | --- | --- | --- |
| FP32 | `artifacts/openvino/fp32/model.xml` | `artifacts/openvino/fp32/model.bin` | Baseline dari checkpoint FP32 final |
| PTQ INT8 | `experiments/ptq/backend_models/ptq_openvino_model/model.xml` | `experiments/ptq/backend_models/ptq_openvino_model/model.bin` | PTQ-only manifest, hash IR, dan source FP32 cocok |
| QAT INT8 | `artifacts/openvino/qat_int8/model.xml` | `artifacts/openvino/qat_int8/model.bin` | True NNCF QAT, winner VAL, direct OpenVINO, numerical equivalence |

SHA256 checkpoint FP32 final: `e05b2b21f7faa68f65671f0aaae37ad962aa5e25c6b8b26853fd0b389d420b8c`. SHA256 checkpoint QAT final: `b9e520cbd5d56597014c932701f471505d2243e02763a2245a51fd83fc35430d`.

| IR | XML SHA256 | BIN SHA256 |
| --- | --- | --- |
| FP32 | `233bb872e218ab3bf525188b0e105eea9f2a79e60ffecb4b767e63939ba257e7` | `e85f45438b5f7c8858b1a168a03f61b12a73b8159391529d19b84e09ff854026` |
| PTQ INT8 | `10cb9dfa3539650a0bc46d60ad708f9ff79fc1e2dc79c5ac4a73ff4943326657` | `82246726edcd780685657f673caa2b40719a8c22b67e3702ca3d1f226b8fcf81` |
| QAT INT8 | `0a97a52f7a72ae9667c6f501ac1e0f44ff04bb5a860ac072e4f39ad406c424b2` | `4a88143019b418edd879932820589c18557dfe9954196fa92025749505bc6352` |

## Rantai provenance

FP32 berasal dari `artifacts/checkpoints/fp32/best.pt`. PTQ dibuat dari FP32 melalui kalibrasi TRAIN historis yang tercatat di `experiments/ptq/manifest.json`; QAT checkpoint dan TEST tidak dipakai. Comparator PTQ yang dibenchmark adalah artifact existing, tanpa PTQ atau kalibrasi baru.

QAT dibuat melalui `nncf.torch.create_compressed_model()` dengan fake quantizer aktif. Kandidat pemenang `qat_lr1e4_r256_e15` dipilih dari VAL, bukan TEST atau metrik PTQ. Checkpoint final cocok dengan manifest dan diekspor melalui direct OpenVINO. Diagnostic export mencatat `PTQ used=false`, `calibration occurred=false`, TEST tidak dimuat, dan equivalence 32/32 citra VAL. Hasil TEST PyTorch berada terpisah di `experiments/final_test/`.

## Periksa setelah clone

Jalankan dari root repository setelah `git lfs pull`:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export TORCH_HOME="$PWD/.cache/torch"
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"

env -u PYTHONPATH -u VIRTUAL_ENV uv run python -c 'import sys; from pathlib import Path; sys.path.insert(0, "scripts"); from op3_model_audit import qat_validation_gate, ptq_validation_gate; r = Path.cwd(); print("QAT:", qat_validation_gate(r)); print("PTQ:", ptq_validation_gate(r, r / "experiments/ptq/backend_models/ptq_openvino_model/model.xml"))'
```

Untuk gate spesifik PTQ/QAT tanpa training, gunakan fungsi `ptq_validation_gate()` dan `qat_validation_gate()` di `scripts/op3_model_audit.py`. Gate harus `accepted: true` dan `reasons: []`. Hash di tabel dapat dicek memakai `sha256sum` pada keenam file IR. File Git LFS sekitar 100–200 byte berarti pointer belum materialized; jangan gunakan untuk inference.

Gate QAT menerima diagnostic direct OpenVINO yang final, sambil tetap mencatat diagnostic ONNX lama sebagai rejected. Menghapus file model kandidat gagal tidak mengubah manifest, hash, atau gate model final.
