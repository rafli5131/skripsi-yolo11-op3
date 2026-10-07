| mode | model | inference_mean_ms | inference_p95_ms | pipeline_mean_ms | effective_fps | system_cpu_mean_percent | process_cpu_mean_percent | system_memory_mean_mb | rss_mean_mb | rss_peak_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | FP32 PyTorch (tanpa OpenVINO) | 52.7003 | 54.3355 | 55.3120 | 18.0694 | 49.8659 | 800.2293 | 4285.1645 | 375.5225 | 391.3851 |
| synthetic | FP32 OpenVINO | 28.0504 | 29.6411 | 29.8745 | 33.4559 | 55.2955 | 893.2841 | 4174.0696 | 240.7455 | 240.8325 |
| synthetic | PTQ INT8 OpenVINO | 11.5404 | 11.8309 | 13.1660 | 75.8624 | 20.5895 | 343.8316 | 4155.2372 | 230.6080 | 230.6908 |
| synthetic | QAT INT8 OpenVINO | 11.2265 | 11.4352 | 12.9337 | 77.2287 | 20.3842 | 341.1632 | 4137.5352 | 225.3613 | 225.4029 |
