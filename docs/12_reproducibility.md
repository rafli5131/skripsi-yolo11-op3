# Reproducibility

## Tujuan

Menetapkan bukti dan batas reproduksibilitas eksperimen.

## Status

**PARTIAL** — konfigurasi, lockfile, seed, SHA256, manifest, dan CSV tersedia.

Seed proyek adalah 42; versi paket dikunci oleh `uv.lock`; metadata dataset mencatat workspace/project tetapi bukan versi Roboflow; model final memiliki SHA256 FP32 `e05b...20b8c` dan QAT best `f969...3a8c8`. Manifest dan artifact freeze QAT menyimpan config serta state eksternal yang dibutuhkan restore. W&B bersifat opt-in di script dan run final dicatat pada manifest.

Keterbatasan: shared GPU server, sumber Roboflow eksternal, satu koreksi polygon lokal, NAS filesystem, dan perilaku NNCF yang spesifik versi. Simpan `.env` di luar Git; gunakan `.env.example` hanya sebagai daftar nama variabel.

## Artefak terkait

[`uv.lock`](../uv.lock), [FP32 SHA](../artifacts/checkpoints/fp32/SHA256SUMS.txt), [QAT SHA](../artifacts/checkpoints/qat/SHA256SUMS.txt), [environment](01_environment.md).
