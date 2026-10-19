# Hasil QAT ROS dengan gerak didampingi operator

## 5. ROS QAT INT8 — robot diam versus berjalan

Tiga run berjalan baru telah selesai pada 19 Oktober 2026. Kondisi gerak dikonfirmasi operator sebelum sesi; setelah sesi operator menyatakan robot tidak jatuh dan sudah berhenti. Robot didampingi/ditopang operator; jumlah penopangan dan kontak kaki dengan lantai tidak didokumentasikan eksplisit. Hasil ini merupakan pengujian inferensi saat robot bergerak dengan pendampingan, bukan validasi gait berjalan mandiri. Konfirmasi terpisah setiap run dan video tidak tersedia.

Seluruh run berjalan memiliki `completed=true`, durasi aktual ≥60 detik, `thermal_abort=false`, dan nol exception termasuk warmup. Pemeriksaan raw CSV sesuai jumlah frame/resource pada JSON. Warmup 30 inference + 30 frame kamera per run, C920 YUYV 640×480, input 640×640/batch 1, CPU, confidence 0,25, IoU 0,45, sampling 0,2 detik, BEST_EFFORT depth 5, debug image/telemetry dimatikan. Kamera kontinu selama tiga run; detector baru untuk setiap run. Domain vision 73 dipisahkan dari kontrol domain 0.

| Kondisi / run | Frame | Durasi (s) | Inference mean / P95 (ms) | Pipeline mean / P95 (ms) | Effective FPS | Exception |
|---|---:|---:|---:|---:|---:|---:|
| Berjalan run_01 | 566 | 60,0197 | 13,0494 / 14,8944 | 16,8577 / 19,2927 | 9,4302 | 0 |
| Berjalan run_02 | 603 | 60,0334 | 12,7908 / 14,7780 | 16,4298 / 19,1687 | 10,0444 | 0 |
| Berjalan run_03 | 638 | 60,0090 | 13,0526 / 14,9911 | 16,7970 / 19,9950 | 10,6317 | 0 |
| Diam gabungan | 2442 | 180,0119 | 13,4190 / 14,9848 | 17,9763 / 20,6444 | 13,5658 | 0 |
| Berjalan gabungan | 1807 | 180,0622 | 12,9642 / 14,8962 | 16,6935 / 19,3917 | 10,0354 | 0 |

### Resource berjalan per run dan perbandingan gabungan

| Kondisi / run | CPU sistem / proses mean (%) | RAM sistem mean (MB) | RSS mean / peak (MB) | Suhu mean / maksimum (°C) | Sampel |
|---|---:|---:|---:|---:|---:|
| run_01 | 6,6476 / 59,0855 | 3458,5560 | 256,8211 / 256,8479 | 44,6351 / 54,0000 | 296 |
| run_02 | 6,6530 / 61,3828 | 3454,2332 | 257,1934 / 257,2165 | 44,8480 / 52,0000 | 296 |
| run_03 | 6,8159 / 66,0156 | 3453,8588 | 257,3326 / 257,3476 | 45,1186 / 51,0000 | 295 |
| diam gabungan | 6,4163 / 88,6172 | 4083,7804 | 257,5252 / 257,8186 | 44,7051 / 56,0000 | 885 |
| berjalan gabungan | 6,7054 / 62,1569 | 3455,5512 | 257,1154 / 257,3476 | 44,8670 / 54,0000 | 887 |

### Variasi antar tiga run berjalan

| Metrik | Rata-rata nilai per run | SD sampel antar-run |
|---|---:|---:|
| Frame | 602,3333 | 36,0046 |
| Effective FPS | 10,0355 | 0,6008 |
| Mean inference (ms) | 12,9643 | 0,1502 |
| Mean pipeline (ms) | 16,6949 | 0,2315 |
| CPU sistem mean (%) | 6,7055 | 0,0956 |
| CPU detector mean (%) | 62,1613 | 3,5300 |
| RAM sistem mean (MB) | 3455,5493 | 2,6106 |
| RSS detector mean (MB) | 257,1157 | 0,2645 |
| Suhu mean (°C) | 44,8673 | 0,2423 |

FPS gabungan diam 13,5658 dan berjalan 10,0354; perubahan relatif -26,0239%. Mean inference berjalan 12,9642 ms dan pipeline 16,6935 ms. FPS berjalan lebih rendah meskipun latency frame yang diproses tidak meningkat. Pengiriman frame sebelum callback, adegan kamera dan beban sistem tidak dikendalikan sepenuhnya; data ini tidak membuktikan bahwa gerak robot sendiri menyebabkan penurunan FPS. CPU proses yang lebih rendah dapat sejalan dengan lebih sedikit frame yang diproses, tetapi penyebabnya tidak diuji lebih lanjut.

Perbandingan ini bersifat deskriptif. Parameter utama terverifikasi cocok melalui protocol.json dan setup JSON, serta format kamera melalui log. Namun tanggal/commit, adegan dan beban kontrol berbeda; QoS/domain lama tidak dicatat eksplisit dan identitas model lama tidak diverifikasi. Perbandingan tidak dinyatakan sepenuhnya setara atau sebagai efek kausal gerakan robot.

Run berjalan lama 12,5234 detik (`final_qat`, 171 frame, `completed=false`) dan 16,6991 detik (`20261019T092114...`, 216 frame, `completed=false`) tetap dikecualikan. Sesi `20261019T092437...` tetap arsip; tidak dicampurkan ke tiga run baru. Metadata commit historis tidak diubah.

Sumber: [raw CSV/JSON dan log](..), [rekap JSON](qat_diam_vs_berjalan_recap.json), [tabel perbandingan CSV](qat_diam_vs_berjalan.csv), [konfirmasi sebelum sesi](../operator_confirmation_start.json), [konfirmasi sesudah sesi](../operator_confirmation_end.json).

## Keterbatasan tambahan

- Operator confirmed walking before session and no fall/stopped after session; separate per-run attestations and video were not recorded.
- Robot accompanied/held by operator; exact support and floor contact not explicitly documented. Results concern inference during operator-supported movement, not validated autonomous gait.
- Reference diam was measured 7 October, walking 19 October. Scene, lighting, software commit and control/system background load are not fully controlled.
- Legacy diam QoS and ROS domain were not explicitly recorded; BEST_EFFORT/depth 5 is inferred from legacy sensor-data QoS. New walking explicitly uses BEST_EFFORT/5 on vision domain 73, controller domain 0.
- Historical model payload identity was not verified at measurement time; current checksums cannot retroactively prove old model identity.
- No motion gateway/fall-detector process was active in this control session; safety observation was by operator. No claim of automatic fall monitoring is made.
- CPU process/RSS measure detector only, not camera plus manager. System RAM includes other processes. MB denotes decimal bytes/1e6.
- FPS is measured successful frames/actual duration; dropped frames before callback are not counted as exceptions. Frame age after publish is not pure transport latency.
- No live-camera ground truth: no precision/recall/F1/mAP is derived from detections.
