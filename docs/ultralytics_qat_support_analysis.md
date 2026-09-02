# Analisis Dukungan QAT Native Ultralytics 8.4.130

## Tujuan

Menentukan dari source package terpasang apakah Ultralytics 8.4.130 menyediakan QAT native untuk training YOLO11.

## Status

**INSPECTED** — tidak ada workload GPU, training, atau export dijalankan.

## Kesimpulan utama

**Ultralytics 8.4.130 has no first-class native YOLO11 QAT training API in this installation.**

Pencarian rekursif source Python di `.venv/lib/python3.11/site-packages/ultralytics/` tidak menemukan modul QAT, `fake_quant`, `enable_qat`, `enable_qat_eager`, atau `prepare_qat`. Tidak ada module `ultralytics.nn.quant`; akibatnya `ultralytics.nn.quant.eager_build` juga tidak ada.

## Trainer dan train arguments

`ultralytics/engine/trainer.py` dan `ultralytics/models/yolo/detect/train.py` tidak memiliki QAT-specific model preparation. Lifecycle trainer hanya menangani model device, parameter trainable, optimizer, AMP, EMA, dan loss; tidak ada penyisipan fake quantizer atau pemanggilan quantization API.

`ultralytics/cfg/default.yaml` memiliki `quantize`, tetapi komentarnya membatasi argumen ini untuk predict/val dan export. Ini bukan `qat=True`, tidak mengubah training lifecycle, dan tidak ditemukan argumen training `int8=True` yang mengaktifkan QAT.

## NNCF dan OpenVINO

NNCF ada hanya di jalur export. `engine/exporter.py:1199-1212` membuat `nncf.Dataset` dari calibration dataloader jika `quantize == 8`. Di `utils/export/openvino.py`, `torch2openvino()` memanggil `nncf.quantize()` hanya pada cabang `quantize == 8`; itu calibrated PTQ, bukan training-time QAT.

Plain OpenVINO export dengan `quantize=None` melakukan trace PyTorch lalu `ov.convert_model` tanpa NNCF PTQ. `nn/backends/openvino.py` mengenali op `FakeQuantize` untuk workaround backend, tetapi tidak menyediakan jalur QAT export/training khusus.

## NVIDIA ModelOpt

ModelOpt terintegrasi hanya pada ONNX/TensorRT export melalui `ultralytics/utils/export/engine.py`:

```python
modelopt_quantize_onnx(onnx_file, quantize=None, dataset=None,
                       shape=(1, 3, 640, 640), dynamic=False, prefix="")
```

Untuk `quantize == 8`, fungsi itu mensyaratkan calibration dataset dan membuat ONNX INT8 Q/DQ. Ini export/PTQ-oriented, bukan NVIDIA ModelOpt QAT integration pada `DetectionTrainer`.

## API/source yang diperiksa

| Item | Hasil |
|---|---|
| `ultralytics.nn.quant` | Tidak ada |
| `ultralytics.nn.quant.eager_build` | Tidak ada |
| `enable_qat_eager()` atau equivalent | Tidak ada |
| QAT-specific BaseTrainer/DetectionTrainer preparation | Tidak ada |
| NNCF training-time integration | Tidak ada |
| NNCF PTQ OpenVINO export | Ada, hanya `quantize == 8` |
| NVIDIA ModelOpt training QAT | Tidak ada |
| NVIDIA ModelOpt ONNX export quantization | Ada |

Signature helper OpenVINO yang terpasang:

```python
torch2openvino(model, im, output_dir=None, dynamic=False,
               quantize=None, calibration_dataset=None,
               int8_detect=False, prefix="")
```

## Implikasi proyek

Integrasi custom NNCF `create_compressed_model()` dengan `DetectionTrainer` diperlukan untuk true QAT proyek ini. Jalur Ultralytics `quantize=8` tidak boleh menggantikannya karena merupakan PTQ berkalibrasi. Lihat [desain QAT](04_qat_design.md), [implementasi QAT](05_qat_implementation.md), dan [OpenVINO export](09_openvino_export.md).
