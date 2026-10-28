"""Readable magnitude and detail views without truncating absolute bars."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

COLORS = ['#285B89', '#B66A20', '#757F37', '#A64A7B', '#666666']


def configure():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 11,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'figure.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.facecolor': 'white',
    })


def local_limits(values, bounds=None):
    """Focus all finite observations with padding, including flat series."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        raise ValueError('No finite observations for the axis')
    lo, hi = float(a.min()), float(a.max())
    span = max(hi - lo, max(abs(lo), abs(hi), 1.0) * 0.0001)
    low, high = lo - span * .30, hi + span * .30
    if bounds:
        low, high = max(low, bounds[0]), min(high, bounds[1])
    if low >= high:
        raise ValueError('Invalid axis bounds')
    return low, high


def style_axis(ax, direction='y'):
    ax.grid(axis=direction, alpha=.22)
    ax.set_axisbelow(True)
    axis = ax.yaxis if direction == 'y' else ax.xaxis
    axis.set_major_locator(MaxNLocator(nbins=5))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)


def comparison(overview, detail, categories, series, unit, decimals=3,
               bounds=None):
    """Zero-based bars plus a clearly labelled focused dot view of identical data.

    series is an ordered mapping of names to values, one per category.
    Every detail panel shares one scale for the full comparison.
    """
    values = np.asarray(list(series.values()), dtype=float)
    if values.shape != (len(series), len(categories)) or not np.isfinite(values).all():
        raise ValueError('Complete finite values required for every series/category')
    if (values < 0).any():
        raise ValueError('Absolute magnitude comparisons require nonnegative values')
    x = np.arange(len(categories))
    width = .72 / len(series)
    if len(series) == 2:
        detail.hlines(x, np.min(values, axis=0), np.max(values, axis=0), color='#AAAAAA', linewidth=1.4, zorder=1)
    for i, (name, ys) in enumerate(series.items()):
        color = COLORS[i % len(COLORS)]
        offset = (i - (len(series) - 1) / 2) * width
        bars = overview.bar(x + offset, ys, width, label=name, color=color)
        # Values are explicit on the focused panel to avoid crowded bar labels.
        yp = x if len(series) == 2 else x + (i - (len(series) - 1) / 2) * .20
        detail.scatter(ys, yp, s=52, color=color,
                       marker=['o', 's', '^', 'D', 'v'][i % 5], label=name, zorder=3, clip_on=False)
        for y, v in zip(yp, ys):
            detail.annotate(f'{v:.{decimals}f}', (v, y), xytext=(0, 8 if i % 2 == 0 else -15),
                            textcoords='offset points', ha='center', fontsize=10)
    overview.set_xticks(x, categories)
    overview.set_ylim(0, max(float(values.max()) * 1.18, 1e-6))
    overview.set_ylabel(unit)
    overview.set_title('Konteks: batang mulai dari nol')
    overview.legend(loc='upper left', bbox_to_anchor=(0, 1.02), frameon=False)
    style_axis(overview)
    low, high = local_limits(values, bounds or (0, float('inf')))
    detail.set_xlim(low, high)
    detail.set_ylim(len(categories) - .45, -.55)
    detail.set_yticks(x, categories)
    detail.set_xlabel(unit)
    detail.set_title(f'Detail titik: rentang {low:.{decimals}f}–{high:.{decimals}f}')
    style_axis(detail, 'x')
    return {'kind': 'magnitude_and_focused_dots', 'connector': 'within_category_aggregate' if len(series) == 2 else None, 'unit': unit,
            'overview_min': 0, 'focus_limits': [low, high],
            'categories': list(categories),
            'series': {k: [float(v) for v in vs] for k, vs in series.items()}}


def delta_plot(ax, categories, baseline, candidate, unit, decimals=4):
    delta = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    y = np.arange(len(categories))
    for i, d in enumerate(delta):
        ax.barh(i, d, color=COLORS[0] if d >= 0 else COLORS[1])
        ax.annotate(f'{d:+.{decimals}f}', (d, i),
                    xytext=(6 if d >= 0 else -6, 0), textcoords='offset points',
                    va='center', ha='left' if d >= 0 else 'right', fontsize=11)
    extent = max(float(np.abs(delta).max()), .001)
    ax.set_xlim(-extent * 1.55, extent * 1.55)
    ax.set_yticks(y, categories)
    ax.invert_yaxis()
    ax.axvline(0, color='#333333', linewidth=1)
    ax.set_xlabel(unit)
    ax.set_title('Selisih QAT − FP32 (tanda dipertahankan)')
    style_axis(ax, 'x')
    return delta.tolist()
