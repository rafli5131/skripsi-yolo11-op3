# Lampiran grafik eksperimen PTQ INT8

Grafik pembanding tambahan untuk benchmark standalone NUC12 dan CPU server. Perbandingan utama tersedia di [galeri grafik](../README.md).

Lihat [lampiran hasil PTQ](../../22_ptq_comparison.md) untuk tabel dan sumber data.

| Grafik | Sumber |
| --- | --- |
| [standalone_synthetic_inference_mean_ms](standalone_synthetic_inference_mean_ms.png) | [experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv](../../../experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv) |
| [standalone_synthetic_effective_fps](standalone_synthetic_effective_fps.png) | [experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv](../../../experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv) |
| [standalone_synthetic_process_cpu_mean_percent](standalone_synthetic_process_cpu_mean_percent.png) | [experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv](../../../experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv) |
| [standalone_synthetic_rss_mean_mb](standalone_synthetic_rss_mean_mb.png) | [experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv](../../../experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv) |
| [standalone_camera_inference_mean_ms](standalone_camera_inference_mean_ms.png) | [experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv](../../../experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv) |
| [standalone_camera_effective_fps](standalone_camera_effective_fps.png) | [experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv](../../../experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv) |
| [standalone_camera_process_cpu_mean_percent](standalone_camera_process_cpu_mean_percent.png) | [experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv](../../../experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv) |
| [standalone_camera_rss_mean_mb](standalone_camera_rss_mean_mb.png) | [experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv](../../../experiments/hardware_op3/comparison_models/20261007_four_models_run1/comparison_all.csv) |
| [server_latency_distribution](server_latency_distribution.png) | [experiments/server_openvino/server_benchmark_fp32.csv](../../../experiments/server_openvino/server_benchmark_fp32.csv)<br>[experiments/server_openvino/server_benchmark_ptq_int8.csv](../../../experiments/server_openvino/server_benchmark_ptq_int8.csv)<br>[experiments/server_openvino/server_benchmark_qat_int8.csv](../../../experiments/server_openvino/server_benchmark_qat_int8.csv) |
| [server_latency_ecdf](server_latency_ecdf.png) | [experiments/server_openvino/server_benchmark_fp32.csv](../../../experiments/server_openvino/server_benchmark_fp32.csv)<br>[experiments/server_openvino/server_benchmark_ptq_int8.csv](../../../experiments/server_openvino/server_benchmark_ptq_int8.csv)<br>[experiments/server_openvino/server_benchmark_qat_int8.csv](../../../experiments/server_openvino/server_benchmark_qat_int8.csv) |
