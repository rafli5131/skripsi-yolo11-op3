| mode | model | inference_mean_ms | inference_p95_ms | pipeline_mean_ms | effective_fps | system_cpu_mean_percent | process_cpu_mean_percent | system_memory_mean_mb | rss_mean_mb | rss_peak_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camera | FP32 PyTorch (tanpa OpenVINO) | 50.9233 | 52.9269 | 56.8538 | 17.5780 | 49.3805 | 783.5617 | 4310.9365 | 388.9649 | 407.6462 |
| camera | FP32 OpenVINO | 37.1724 | 59.8114 | 48.4690 | 20.6250 | 44.8789 | 706.8708 | 4152.4328 | 242.7902 | 243.3475 |
| camera | PTQ INT8 OpenVINO | 12.5824 | 15.3498 | 43.3076 | 23.0824 | 8.4631 | 127.3879 | 4154.7287 | 233.9171 | 234.5247 |
| camera | QAT INT8 OpenVINO | 12.0950 | 14.8793 | 41.1139 | 24.3139 | 8.5440 | 128.3852 | 4142.1401 | 228.3978 | 229.0237 |
