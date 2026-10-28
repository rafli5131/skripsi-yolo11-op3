#!/usr/bin/env python3
"""Regenerate readable thesis figures from frozen CSV/JSON, without model execution."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.reporting.figure_style import (
    COLORS, comparison, configure, delta_plot, local_limits, plt, style_axis,
)

FP = 'experiments/fp32/fp32_baseline/results.csv'
SEARCH = 'artifacts/checkpoints/qat/selection_manifest.json'
TEST = 'experiments/final_test/'
HARDWARE = 'experiments/hardware_op3/'
ROS = HARDWARE + 'operator_runs/20261019T132747Z_fp32_berjalan_final/analysis/'
STABILITY = HARDWARE + 'stability_qat/20261007_ros_qat/'
MODELS = ['FP32 PyTorch\ntanpa OpenVINO', 'FP32\nOpenVINO',
          'PTQ INT8\nOpenVINO', 'QAT INT8\nOpenVINO']
METRICS = ['precision', 'recall', 'F1', 'mAP50', 'mAP50-95']


class Figures:
    def __init__(self, root, output, formats):
        self.root, self.output, self.formats = root, output, formats
        self.inventory = []
        output.mkdir(parents=True, exist_ok=True)

    def rows(self, path):
        with (self.root / path).open(newline='') as f:
            return [{k.strip(): v.strip() for k, v in row.items()}
                    for row in csv.DictReader(f)]

    def data(self, path):
        return json.loads((self.root / path).read_text())

    def save(self, fig, name, sources, note, metadata=None):
        # Reserve title/footer space explicitly rather than squeezing plot labels.
        fig.subplots_adjust(top=.83, bottom=.23, left=.09, right=.96, wspace=.48, hspace=.65)
        fig.text(.5, .035, note, ha='center', va='bottom', fontsize=10, wrap=True)
        for extension in self.formats:
            buffer = io.BytesIO()
            fig.savefig(buffer, format=extension, dpi=180,
                        metadata={'Creator': 'frozen-results plotting script'}
                        if extension == 'svg' else None)
            payload = buffer.getvalue()
            if extension == 'png':
                from PIL import Image
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
            target = self.output / f'{name}.{extension}'
            temporary = target.with_suffix(target.suffix + '.tmp')
            temporary.write_bytes(payload)
            os.replace(temporary, target)
        self.inventory.append({'figure': name, 'sources': sources,
                               'note': note, **(metadata or {})})
        plt.close(fig)

    def compare(self, name, title, categories, series, unit, sources, note,
                decimals=3, bounds=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
        fig.suptitle(title, fontsize=16)
        metadata = comparison(*axes, categories, series, unit, decimals, bounds)
        self.save(fig, name, sources,
                  note + '\nPanel kanan memakai posisi titik dan skala lokal, bukan panjang batang.', metadata)

    def training(self, rows, prefix, title, source, tail=False):
        x = np.array([float(r['epoch']) for r in rows])
        mask = x >= max(1, x.max() - 19) if tail else np.ones(len(x), dtype=bool)
        fields = ['metrics/precision(B)', 'metrics/recall(B)',
                  'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        best = max(rows, key=lambda r: float(r['metrics/mAP50-95(B)']))
        limits = {}
        for ax, field, label in zip(axes.flat, fields, ['Precision', 'Recall', 'mAP50', 'mAP50–95']):
            y = np.array([float(r[field]) * 100 for r in rows])
            ax.plot(x[mask], y[mask], color=COLORS[0], marker='o', markersize=3)
            limits[label] = local_limits(y[mask], (0, 100))
            ax.set_ylim(*limits[label])
            if mask[np.where(x == float(best['epoch']))[0][0]]:
                ax.axvline(float(best['epoch']), color='#555555', linestyle='--', linewidth=1)
            ax.set_title(label)
            ax.set_ylabel('Skor (%)')
            ax.set_xlabel('Epoch')
            style_axis(ax)
            ticks = np.unique(np.r_[x[mask][0], x[mask][-1],
                                     np.linspace(x[mask][0], x[mask][-1], 5).round()])
            ax.set_xticks(ticks)
            ax.set_xlim(x[mask][0] - .2, x[mask][-1] + .2)
        fig.suptitle(title, fontsize=16)
        self.save(fig, prefix, [source],
                  'Skala y lokal dan berbeda per metrik; semua observasi dalam rentang ditampilkan.\n'
                  f"Garis putus-putus: best mAP50–95 VAL, epoch {best['epoch']}. Bukan evaluasi TEST.",
                  {'kind': 'training_metrics', 'limits': limits, 'epochs_displayed': int(mask.sum())})

    def losses(self, rows, prefix, title, source):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
        x = [float(r['epoch']) for r in rows]
        for ax, field in zip(axes, ['box', 'cls', 'dfl']):
            all_values = []
            for i, split in enumerate(['train', 'val']):
                y = [float(r[f'{split}/{field}_loss']) for r in rows]
                all_values.extend(y)
                ax.plot(x, y, color=COLORS[i], linestyle=['-', '--'][i], label=split.upper())
            ax.set_ylim(*local_limits(all_values))
            ax.set_title(field.upper())
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend(frameon=False)
            style_axis(ax)
        fig.suptitle(title, fontsize=16)
        self.save(fig, prefix, [source],
                  'Skala lokal tiap komponen; TRAIN dan VAL berbagi skala dalam panel yang sama.\n'
                  'Seluruh epoch ditampilkan tanpa smoothing. Nilai loss antarkomponen tidak dijumlahkan.')

    def build(self):
        configure()
        selection = self.data(SEARCH)
        candidates = selection['candidates']
        labels = [f'C{i+1}' for i in range(len(candidates))]
        best = [c['validation']['mAP50-95'] * 100 for c in candidates]
        self.compare('qat_candidate_best', 'Seleksi kandidat QAT pada VAL', labels,
                     {'mAP50–95': best}, 'mAP50–95 (%)', [SEARCH],
                     f'C4 − C3 = {best[3]-best[2]:+.3f} poin persentase; seleksi memakai VAL.',
                     3, (0, 100))
        fig, ax = plt.subplots(figsize=(12, 5.8))
        sources = []
        for i, candidate in enumerate(candidates):
            source = f"artifacts/checkpoints/qat_search/{candidate['name']}/results.csv"
            rows = self.rows(source)
            sources.append(source)
            ax.plot([float(r['epoch']) for r in rows],
                    [float(r['metrics/mAP50-95(B)']) * 100 for r in rows],
                    color=COLORS[i], marker=['o', 's', '^', 'D', 'v'][i], label=labels[i])
            self.losses(rows, f'qat_c{i+1}_loss', f'Loss QAT kandidat C{i+1}', source)
            self.training(rows, f'qat_c{i+1}_metrics', f'Metrik VAL QAT kandidat C{i+1}', source)
        ax.set_ylabel('mAP50–95 VAL (%)')
        ax.set_xlabel('Epoch')
        ax.legend(ncol=5, frameon=False)
        style_axis(ax)
        fig.suptitle('Perkembangan lima kandidat QAT', fontsize=16)
        self.save(fig, 'qat_candidate_history', sources,
                  'Sumbu y lokal untuk melihat fluktuasi. Panjang garis mengikuti jumlah epoch aktual.\n'
                  'C4 terpilih pada epoch 5; TEST tidak dipakai untuk seleksi.')
        rows = self.rows(FP)
        self.losses(rows, 'fp32_loss', 'Loss baseline FP32, 100 epoch', FP)
        self.training(rows, 'fp32_metrics', 'Metrik VAL baseline FP32', FP)
        self.training(rows, 'fp32_metrics_last20', 'Detail 20 epoch terakhir FP32', FP, tail=True)

        a, b = [self.data(TEST + k + '_test_metrics.json') for k in ['fp32', 'qat']]
        sources = [TEST + k + '_test_metrics.json' for k in ['fp32', 'qat']]
        av = np.array([a['overall'][k] * 100 for k in METRICS])
        bv = np.array([b['overall'][k] * 100 for k in METRICS])
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
        x = np.arange(len(METRICS))
        axes[0].bar(x - .18, av, .36, color=COLORS[0], label='FP32')
        axes[0].bar(x + .18, bv, .36, color=COLORS[1], label='QAT')
        axes[0].set_xticks(x, METRICS)
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel('Skor (%)')
        axes[0].set_title('Konteks penuh: batang mulai dari nol')
        axes[0].legend(frameon=False)
        style_axis(axes[0])
        delta = delta_plot(axes[1], METRICS, av, bv, 'Selisih (poin persentase)')
        fig.suptitle('Kualitas deteksi pada TEST yang sama', fontsize=16)
        self.save(fig, 'test_overall_and_delta', sources,
                  '592 citra, 734 anotasi; checkpoint PyTorch. Selisih positif berarti skor naik.\n'
                  'Perubahan kecil tetap kecil secara numerik; tidak menunjukkan uji signifikansi.',
                  {'kind': 'overall_and_delta', 'delta_pp': delta, 'fp32_percent': av.tolist(), 'qat_percent': bv.tolist()})
        fig, axes = plt.subplots(1, 5, figsize=(16, 5.8))
        for ax, metric in zip(axes, METRICS):
            vals = [a['overall'][metric] * 100, b['overall'][metric] * 100]
            for i, val in enumerate(vals):
                ax.scatter(i, val, color=COLORS[i], marker=['o', 's'][i], s=65)
                ax.annotate(f'{val:.4f}', (i, val), xytext=(0, 10),
                            textcoords='offset points', ha='center', fontsize=10)
            ax.set_xticks([0, 1], ['FP32', 'QAT'])
            ax.set_xlim(-.55, 1.55)
            ax.set_ylim(*local_limits(vals, (0, 100)))
            ax.set_title(metric)
            ax.set_ylabel('Skor (%)')
            style_axis(ax)
        fig.suptitle('Detail setiap metrik TEST', fontsize=16)
        self.save(fig, 'test_overall_focused', sources,
                  'Skala y lokal berbeda per panel; baca nilai dan tick setiap metrik.\n'
                  'Titik membandingkan posisi skor, bukan panjang batang. Tidak ada error bar karena replikasi training tidak tersedia.')
        for metric in METRICS:
            classes = ['ball', 'gawang', 'robot']
            self.compare('test_class_' + metric.replace('-', '_'), f'{metric} per kelas pada TEST',
                         classes, {'FP32': [a['per_class'][c][metric] * 100 for c in classes],
                                   'QAT': [b['per_class'][c][metric] * 100 for c in classes]},
                         'Skor (%)', sources, 'Checkpoint PyTorch; split TEST yang sama. Perubahan tiap kelas dapat berbeda.',
                         3, (0, 100))

        ros_source = ROS + 'ros_fp32_vs_qat_conditions.csv'
        ros = {r['dataset']: r for r in self.rows(ros_source)}
        keys = ['inference_mean_ms', 'inference_p95_ms', 'pipeline_mean_ms', 'effective_fps',
                'process_cpu_percent_mean', 'system_cpu_percent_mean', 'detector_rss_mb_mean', 'system_ram_mb_mean']
        titles = ['Mean inferensi', 'P95 inferensi', 'Mean pipeline', 'FPS efektif',
                  'CPU proses detector', 'CPU sistem', 'RSS proses detector', 'RAM sistem']
        units = ['ms', 'ms', 'ms', 'frame/detik', '% multi-core', '%', 'MB desimal', 'MB desimal']
        for key, title, unit in zip(keys, titles, units):
            self.compare('ros_' + key, title + ' ROS 2 final', ['Diam', 'Gerak\ndidampingi'],
                         {'FP32': [float(ros[g][key]) for g in ['fp32_diam', 'fp32_berjalan']],
                          'QAT': [float(ros[g][key]) for g in ['qat_diam_20261019', 'qat_berjalan']]},
                         unit, [ros_source], '3 × sekitar 60 detik/model/kondisi, BEST_EFFORT depth 5.\n'
                         'Frame terproses berbeda; CPU/RSS bukan energi per frame atau seluruh RAM robot.', 3)
        runs_source = ROS + 'per_run_recap.csv'
        runs = self.rows(runs_source)
        for key, title, unit in [('effective_fps', 'FPS', 'frame/detik'),
                                 ('inference_mean_ms', 'Mean inferensi', 'ms')]:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
            for ax, suffix, condition in zip(axes, ['diam', 'berjalan'], ['Diam', 'Gerak didampingi']):
                vals = []
                for i, model in enumerate(['fp32', 'qat']):
                    group = f'{model}_{suffix}' + ('_20261019' if model == 'qat' and suffix == 'diam' else '')
                    group_rows = sorted((r for r in runs if r['dataset'] == group), key=lambda r: r['run_id'])
                    y = [float(r[key]) for r in group_rows]
                    vals.extend(y)
                    ax.plot(range(1, len(y)+1), y, color=COLORS[i], marker=['o', 's'][i], label=model.upper())
                ax.set_ylim(*local_limits(vals))
                ax.set_xticks([1, 2, 3])
                ax.set_xlabel('Run')
                ax.set_ylabel(unit)
                ax.set_title(condition)
                ax.legend(frameon=False)
                style_axis(ax)
            fig.suptitle(title + ' pada tiga run ROS', fontsize=16)
            self.save(fig, 'ros_runs_' + key, [runs_source],
                      'Skala lokal tiap kondisi; pengulangan sesi, bukan replikasi seed training.\n'
                      'Semua tiga run ditampilkan tanpa menyimpulkan signifikansi statistik.')

        source = HARDWARE + 'comparison_models/20261007_four_models_run1/comparison_all.csv'
        all_rows = self.rows(source)
        for mode in ['synthetic', 'camera']:
            subset = [r for r in all_rows if r['mode'] == mode]
            for key, title, unit in [('inference_mean_ms', 'Mean inferensi', 'ms'),
                                     ('effective_fps', 'FPS efektif', 'frame/detik'),
                                     ('process_cpu_mean_percent', 'CPU proses', '% multi-core'),
                                     ('rss_mean_mb', 'RSS proses', 'MB desimal')]:
                self.compare(f'standalone_{mode}_{key}', title + f' standalone NUC12 ({mode})',
                             MODELS, {'Nilai terukur': [float(r[key]) for r in subset]}, unit,
                             [source], 'Input model 640×640, batch 1, warm-up 30.\n'
                             'Sintetis dan kamera merupakan protokol berbeda; scene kamera antarrun tidak identik.', 3)
        self.server()
        self.stability()
        self.write_index()

    def server(self):
        sources = [f'experiments/server_openvino/server_benchmark_{k}.csv'
                   for k in ['fp32', 'ptq_int8', 'qat_int8']]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
        for ax, key, title in zip(axes, ['inference_ms', 'loop_wall_ms'], ['Inferensi', 'Loop standalone']):
            samples = [[float(r[key]) for r in self.rows(source)] for source in sources]
            ax.boxplot(samples, showfliers=True)
            ax.set_xticks([1, 2, 3], ['FP32', 'PTQ', 'QAT'])
            ax.set_title(title)
            ax.set_ylabel('ms')
            style_axis(ax)
        fig.suptitle('Distribusi latensi CPU server', fontsize=16)
        self.save(fig, 'server_latency_distribution', sources,
                  '300 iterasi/model, 30 warm-up; median dan sebaran dari CSV asli, outlier tetap terlihat.\n'
                  'Server AMD EPYC 7413 adalah diagnostik host, bukan latensi robot.')

    def stability(self):
        source = STABILITY + 'latency/stability_run_01_frames.csv'
        rows = self.rows(source)
        start = datetime.fromisoformat(rows[0]['timestamp'])
        x = np.array([(datetime.fromisoformat(r['timestamp']) - start).total_seconds() for r in rows])
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        for ax, field, title in zip(axes, ['inference_ms', 'latency_ms'], ['Inferensi', 'Pipeline']):
            y = np.array([float(r[field]) for r in rows])
            ax.plot(x, y, color='#BBBBBB', linewidth=.45, label='Setiap frame')
            bins = np.floor(x / 5).astype(int)
            ids = np.unique(bins)
            ax.plot([x[bins == i].mean() for i in ids], [y[bins == i].mean() for i in ids],
                    color=COLORS[0], linewidth=2, label='Mean 5 detik')
            ax.set_ylabel(title + ' (ms)')
            ax.set_ylim(*local_limits(y, (0, float('inf'))))
            ax.legend(frameon=False)
            style_axis(ax)
        axes[-1].set_xlabel('Detik sejak frame terukur pertama')
        fig.suptitle('Latensi selama stabilitas QAT', fontsize=16)
        self.save(fig, 'stability_latency', [source],
                  'Semua frame termasuk lonjakan ditampilkan; garis tebal hanya ringkasan jendela.\n'
                  'Skala y lokal per tahap. Sesi sekitar 600 detik, terpisah dari ROS final 3×60 detik.')
        source = STABILITY + 'resources/stability_run_01.csv'
        rows = self.rows(source)
        x = [float(r['elapsed_seconds']) for r in rows]
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        for ax, field, unit in zip(axes, ['detector_process_cpu_percent', 'detector_rss_mb', 'temperature_c'],
                                    ['CPU proses (%)', 'RSS proses (MB desimal)', 'Suhu thermal zone (°C)']):
            y = [float(r[field]) if r[field] else np.nan for r in rows]
            ax.plot(x, y, color=COLORS[0], linewidth=.8)
            ax.set_ylim(*local_limits(y))
            ax.set_ylabel(unit)
            style_axis(ax)
        axes[-1].set_xlabel('Detik terukur')
        fig.suptitle('Sumber daya selama stabilitas QAT', fontsize=16)
        self.save(fig, 'stability_resources', [source],
                  'Sumbu y lokal tiap besaran; semua sampel asli ditampilkan.\n'
                  'Perubahan RSS kecil tetap berlabel MB. Satu run tidak membuktikan ada/tidak ada memory leak.')

    def write_index(self):
        (self.output / 'figure_inventory.json').write_text(json.dumps(self.inventory, indent=2, ensure_ascii=False) + '\n')
        text = ['# Grafik hasil penelitian', '',
                'Dibuat dari CSV/JSON beku, tanpa training, evaluasi ulang, atau pengukuran robot.', '',
                'Batang absolut dimulai dari nol. Panel titik memakai skala lokal yang tertulis. '
                'Skor deteksi ditampilkan dalam persen; selisihnya dalam poin persentase. '
                'Perbaikan dan penurunan sama-sama ditampilkan. Ini pembesaran visual, bukan bukti signifikansi.', '',
                '## Contoh utama', '',
                '![Kandidat QAT: konteks dan detail](qat_candidate_best.png)', '',
                '![TEST: skor dan selisih bertanda](test_overall_and_delta.png)', '',
                '![TEST: pembesaran setiap metrik](test_overall_focused.png)', '',
                '## Seluruh grafik dan sumber', '',
                '| Grafik | Sumber |', '| --- | --- |']
        for item in self.inventory:
            text.append(f"| [{item['figure']}]({item['figure']}.png) | " + '<br>'.join(
                f'[{p}](../../{p})' for p in item['sources']) + ' |')
        text.extend(['', 'Lihat [alasan revisi dan cara menjalankan](../32_revisi_visualisasi_grafik.md).', ''])
        (self.output / 'README.md').write_text('\n'.join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--output', type=Path, default=REPO / 'docs/figures')
    parser.add_argument('--formats', nargs='+', choices=['png', 'svg', 'pdf'], default=['png'])
    args = parser.parse_args()
    builder = Figures(args.repo_root.resolve(), args.output.resolve(), args.formats)
    builder.build()
    print(f'{len(builder.inventory)} figures written to {builder.output}')


if __name__ == '__main__':
    main()
