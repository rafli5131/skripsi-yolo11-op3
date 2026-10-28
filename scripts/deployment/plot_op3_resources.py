#!/usr/bin/env python3
"""Render measured resources on separate, labelled local scales outside timing."""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.reporting.figure_style import configure, local_limits, plt, style_axis


def plot_resources(csv_path, output):
    if output.exists():
        raise SystemExit('Refusing to overwrite evidence image')
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit('No actual resource samples to plot')
    configure()
    t = [float(r['elapsed_seconds']) for r in rows]
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    fields = ['cpu_usage_percent', 'detector_process_cpu_percent', 'memory_used_mb',
              'detector_rss_mb', 'temperature_c']
    units = ['System CPU (%)', 'Detector CPU (%)\nmulti-core', 'System used RAM\n(MB decimal)',
             'Detector RSS\n(MB decimal)', 'Max thermal zone\n(°C)']
    for ax, field, unit in zip(axes, fields, units):
        y = [float(r[field]) if r[field] else float('nan') for r in rows]
        ax.plot(t, y, color='#285B89', linewidth=.8)
        try:
            ax.set_ylim(*local_limits(y, (0, float('inf'))))
        except ValueError:
            ax.text(.5, .5, 'No finite samples', transform=ax.transAxes, ha='center')
        ax.set_ylabel(unit)
        style_axis(ax)
    axes[-1].set_xlabel('Measured elapsed time (seconds)')
    fig.suptitle(f"{rows[0]['condition']} / {rows[0]['run_id']}\nSource: {csv_path.name}")
    fig.subplots_adjust(top=.92, bottom=.08, left=.16, right=.97, hspace=.35)
    fig.text(.5, .015, 'Local y-scale per panel; all measured samples shown. CPU and RAM scopes are separate.\n'
             'A zoomed RSS scale reveals small changes, not evidence of a memory leak.', ha='center', fontsize=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot_resources(args.csv, args.output)


if __name__ == '__main__':
    main()
