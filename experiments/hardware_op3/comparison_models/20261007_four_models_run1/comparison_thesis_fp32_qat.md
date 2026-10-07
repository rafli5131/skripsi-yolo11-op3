| mode | model | inference_mean_ms | inference_p95_ms | pipeline_mean_ms | effective_fps | system_cpu_mean_percent | process_cpu_mean_percent | system_memory_mean_mb | rss_mean_mb | rss_peak_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | FP32 PyTorch (tanpa OpenVINO) | 52.7003 | 54.3355 | 55.3120 | 18.0694 | 49.8659 | 800.2293 | 4285.1645 | 375.5225 | 391.3851 |
| synthetic | FP32 OpenVINO | 28.0504 | 29.6411 | 29.8745 | 33.4559 | 55.2955 | 893.2841 | 4174.0696 | 240.7455 | 240.8325 |
| synthetic | QAT INT8 OpenVINO | 11.2265 | 11.4352 | 12.9337 | 77.2287 | 20.3842 | 341.1632 | 4137.5352 | 225.3613 | 225.4029 |
| camera | FP32 PyTorch (tanpa OpenVINO) | 50.9233 | 52.9269 | 56.8538 | 17.5780 | 49.3805 | 783.5617 | 4310.9365 | 388.9649 | 407.6462 |
| camera | FP32 OpenVINO | 37.1724 | 59.8114 | 48.4690 | 20.6250 | 44.8789 | 706.8708 | 4152.4328 | 242.7902 | 243.3475 |
| camera | QAT INT8 OpenVINO | 12.0950 | 14.8793 | 41.1139 | 24.3139 | 8.5440 | 128.3852 | 4142.1401 | 228.3978 | 229.0237 |
