# Reproducibility

## Tujuan

Menetapkan bukti dan batas reproduksibilitas eksperimen.

## Status

**PARTIAL** — konfigurasi, lockfile, seed, manifest, dan CSV tersedia.

Seed proyek adalah 42; versi paket dikunci oleh `uv.lock`; metadata dataset mencatat workspace/project tetapi bukan versi Roboflow...20b8c` dan QAT best `f969...3a8c8`. Manifest dan artifact freeze QAT menyimpan config serta state eksternal yang dibutuhkan restore. W&B bersifat opt-in di script dan run final dicatat pada manifest.

Keterbatasan: shared GPU server, sumber Roboflow eksternal, satu koreksi polygon lokal, NAS filesystem, dan perilaku NNCF yang spesifik versi. Simpan `.env` di luar Git; gunakan `.env.example` hanya sebagai daftar nama variabel.

## Artefak terkait

[`uv.lock`](../uv.lock), [environment](01_environment.md).
