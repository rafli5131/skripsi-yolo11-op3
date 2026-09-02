# Diagnostik QAT CPU/CUDA

## Tujuan

Memisahkan mismatch backend dari mismatch jalur OpenVINO.

## Status

Phase 7B **FAILED** untuk synthetic lama; Phase 7C **PASSED** untuk input representatif.

Phase 7B memulihkan 1.323 state entries dan mempertahankan 184 quantizer. Satu VAL image memiliki relative MAE 0,715% (**PASS**), sedangkan fixed synthetic lama 29,46% (**FAIL**); output finite dan `[1,7,8400]`. Saat itu kandidat export dihentikan sebelum OpenVINO, sehingga kegagalan bukan bukti bahwa OpenVINO menyebabkan mismatch dan bukan bukti definitif NNCF CPU rusak.

Phase 7C membandingkan raw YOLO output `[1, 7, 8400]` dari restored QAT model yang sama pada CUDA dan CPU.

Synthetic lama adalah `torch.linspace(-1, 1, 1*3*640*640).reshape([1,3,640,640])`:
min -1, max 1, mean mendekati 0, std 0,57735. Input tersebut gagal dan tidak
merepresentasikan domain input YOLO deployment.

Synthetic representatif menggunakan `torch.rand([1,3,640,640])` dengan seed
42 dalam rentang `[0,1]`: min 0,000000238; max 0,999998689; mean 0,499556;
std 0,288615. Ia lolos batas relative MAE 1%.

Tepat 32 gambar VAL dipilih lewat lexical sort dan diproses dengan cv2 decode,
LetterBox 640, BGR->RGB, NCHW float32, lalu `/255`. Statistik agregatnya min
0, max 1, mean 0,354327, std 0,215821. Semua 32 lolos: minimum 0,5971%, median
0,7485%, mean 0,7500%, p95 0,8165%, maksimum 0,8402%.

Kesimpulan resmi Phase 7C: **"Mismatch is likely input-distribution-specific."** CPU
backend bukan lagi export blocker. `torch.randn` atau input out-of-domain tidak
boleh digunakan sebagai referensi equivalence export. TEST tidak digunakan.
