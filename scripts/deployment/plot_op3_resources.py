#!/usr/bin/env python3
"""Render evidence from measured resource CSV, outside the timed experiment."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--csv', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
with args.csv.open(newline='') as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit('No actual resource samples to plot')
t = [float(r['elapsed_seconds']) for r in rows]
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
axes[0].plot(t, [float(r['cpu_usage_percent']) for r in rows], label='System CPU (%)')
axes[0].plot(t, [float(r['detector_process_cpu_percent']) for r in rows], label='Detector CPU (%; multi-core)')
axes[0].set_ylabel('CPU (%)')
axes[1].plot(t, [float(r['memory_used_mb']) for r in rows], label='System used memory (MB)')
axes[1].plot(t, [float(r['detector_rss_mb']) for r in rows], label='Detector RSS (MB)')
axes[1].set_ylabel('Memory (MB decimal)')
axes[2].plot(t, [float(r['temperature_c']) if r['temperature_c'] else float('nan') for r in rows], label='Max thermal zone (C)')
axes[2].set_ylabel('Temperature (C)')
axes[2].set_xlabel('Measured elapsed time (seconds)')
for axis in axes:
    axis.legend()
    axis.grid(alpha=.3)
fig.suptitle(f"{rows[0]['condition']} / {rows[0]['run_id']} — user-supplied QAT OpenVINO folder\nSource: {args.csv.name}")
fig.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
if args.output.exists():
    raise SystemExit('Refusing to overwrite evidence image')
fig.savefig(args.output, dpi=150)
