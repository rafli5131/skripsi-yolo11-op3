# ROS QAT-folder stability test

Actual camera run on stationary robot; user-designated QAT artifact, no hash checks.

{
  "duration_seconds": 600.0161936560007,
  "frames": 8260,
  "fps": 13.766295122253977,
  "latency_mean_ms": 17.6222128279661,
  "exceptions": 0,
  "rss_start_bytes": 257945600,
  "rss_end_bytes": 258379776,
  "rss_growth_bytes": 434176,
  "max_temperature_c": 59.0
}

RSS was measured before CSV summary allocation. The observed growth is not by itself proof of a memory leak or proof that no leak exists. See raw periodic RSS and resource plot. Dropped camera frames are unknown.
