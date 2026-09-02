# Desain Quantization-Aware Training

## Tujuan

Menjelaskan alasan dan mekanisme true QAT yang digunakan.

## Status

**DONE**.

QAT mempertahankan bobot FP32 tetapi menyisipkan fake quantization pada forward pass. Error kuantisasi terlihat saat loss/backpropagation sehingga bobot dapat beradaptasi sebelum target deployment terkuantisasi. PTQ mengkuantisasi setelah training dan tidak memberikan adaptasi gradien tersebut.

Penelitian ini menolak `nncf.quantize()` karena docstring NNCF terpasang mengidentifikasinya sebagai post-training quantization. Jalur terpilih: `NNCFConfig`, `register_default_init_args()`, `nncf.torch.create_compressed_model()`, dan `QuantizationController.init_range(PTRangeInitParams)` menggunakan TRAIN.

## Artefak terkait

[probe QAT](../scripts/probe_qat_compatibility.py), [implementasi](05_qat_implementation.md), [manifest probe](../experiments/debug/qat_compatibility_probe.json).
