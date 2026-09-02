# Phase 7F — NNCF QAT → ONNX → OpenVINO

## Tujuan

Memisahkan transformasi deployment QAT menjadi dua tahap yang dapat diuji secara independen:

```text
QAT PyTorch ter-restore → ONNX → OpenVINO IR
```

Tujuannya adalah mengidentifikasi secara tepat apakah divergensi, bila ada, berasal dari ekspor NNCF ke ONNX atau dari konversi ONNX ke OpenVINO.

## Status

**NOT YET EXECUTED.** Skrip telah disiapkan tetapi tidak dijalankan dari sandbox pengembangan. Hasil runtime akan ditulis ke `artifacts/openvino/qat_onnx_export_diagnostic.json`.

## Dasar keputusan

Jalur historis `restored QAT -> nncf.torch.strip() -> OpenVINO` adalah **REJECTED** karena relative MAE 36,24%, walaupun IR memperlihatkan `FakeQuantize`, konstanta INT8, dan `Convert`. Struktur graph yang tampak terkuantisasi tidak cukup tanpa ekuivalensi numerik.

PTQ dilarang untuk penelitian ini. Secara khusus skrip tidak memanggil `nncf.quantize()`, `nncf.torch.strip()`, `quantize=8`, `int8=True`, atau calibration dataset. TEST juga tidak dibaca.

## API NNCF terpasang

Controller yang dihasilkan oleh `nncf.torch.create_compressed_model()` adalah `nncf.torch.quantization.algo.QuantizationController`. NNCF 2.13.0 terpasang menyediakan API berikut:

```python
compression_ctrl.export_model(
    save_path: str,
    save_format: Optional[str] = None,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    model_args: Optional[Tuple[Any, ...]] = None,
)
```

Implementasinya diteruskan ke `nncf.torch.exporter.PTExporter`, yang mendukung `onnx` dan `onnx_<opset>`. Karena itu jalur resmi ini dipilih; generic `torch.onnx.export()` tidak dipakai sebagai fallback.

## Workflow

1. Restore checkpoint final: FP32 architecture → `NNCFConfig` → `create_compressed_model()` → compression state → model state.
2. Verifikasi 1.323 state entries dan 184 quantizer aktif.
3. Jalankan forward referensi pada instance native dalam `eval()` tanpa `Detect.export`.
4. Restore instance kedua untuk ekspor, aktifkan `Detect.export=True` hanya pada instance ini, lalu panggil `compression_ctrl.export_model(..., save_format="onnx")`.
5. Jalankan `onnx.checker.check_model()` bila paket `onnx` tersedia.
6. Bila ONNX Runtime dapat memuat graph, bandingkan raw output ONNX dengan referensi PyTorch. NNCF 2.13.0 dapat mengekspor custom `org.openvinotoolkit::FakeQuantize`; stock ONNX Runtime tidak memiliki kernel operator tersebut. Kondisi ini dicatat sebagai **NOT EXECUTABLE**, bukan numerical mismatch, lalu OpenVINO ONNX importer menjadi tahap eksekusi berikutnya.
7. Konversi ONNX dengan `openvino.convert_model()`, lalu bandingkan OpenVINO CPU dengan referensi PyTorch.

Seluruh verifikasi menggunakan 32 citra VAL yang dipilih secara lexical deterministik dan diproses dengan `LetterBox(640)`, BGR→RGB, NCHW, `float32`, dan normalisasi `/255`. Raw output yang dibandingkan harus `[1,7,8400]`, bukan post-NMS.

## Kriteria penerimaan

Untuk setiap citra VAL dihitung MAE, magnitudo referensi, relative MAE, galat maksimum, dan cosine similarity. Semua citra harus memenuhi relative MAE <=1%. Setelah itu IR harus memperlihatkan bukti semantik kuantisasi, misalnya `FakeQuantize` dan/atau konstanta I8/U8.

Hanya kandidat yang lulus kedua gate dipromosikan ke:

```text
artifacts/openvino/qat_int8/model.onnx
artifacts/openvino/qat_int8/model.xml
artifacts/openvino/qat_int8/model.bin
artifacts/openvino/qat_int8/SHA256SUMS.txt
```

Jika salah satu gate gagal, kandidat tetap pada `artifacts/openvino/debug/` dan final `qat_int8` tidak ditimpa.

## Ketersediaan dependency

Dependency proyek sekarang mengunci `onnx 1.17.0` dan `onnxruntime 1.20.1`. ONNX Runtime tervalidasi menyediakan `CPUExecutionProvider`. Oleh karena itu run Phase 7F berikutnya dapat memisahkan numerik PyTorch QAT → ONNX dari numerik ONNX → OpenVINO. Skrip tetap tidak menginstal dependency secara otomatis; status ketersediaan juga selalu dicatat ulang saat runtime.

## Artefak terkait

- [`scripts/export_qat_onnx_openvino.py`](../scripts/export_qat_onnx_openvino.py)
- [OpenVINO export](09_openvino_export.md)
- [Analisis kegagalan strip](10_openvino_export_failure_analysis.md)
- [Diagnostik CPU/CUDA](11_cpu_cuda_qat_diagnostic.md)
