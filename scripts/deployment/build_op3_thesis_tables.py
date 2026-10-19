#!/usr/bin/env python3
"""Rebuild thesis tables from completed, measured ROS runs (sample SD, ddof=1)."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def stats(values):
    return {'mean': statistics.mean(values) if values else None,
            'sd': statistics.stdev(values) if len(values) > 1 else None,
            'peak': max(values) if values else None}


def write_table(path, rows, *, markdown=True):
    if not rows:
        return
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if not markdown:
        return
    def cell(value):
        if value is None:
            return 'belum diukur / tidak tersedia'
        return f'{value:.4f}' if isinstance(value, float) else str(value)
    lines = ['| '+' | '.join(fields)+' |', '| '+' | '.join(['---']*len(fields))+' |']
    lines += ['| '+' | '.join(cell(row[k]) for k in fields)+' |' for row in rows]
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def build(root):
    summaries, samples, reports = [], [], []
    for path in sorted((root/'run_metadata').glob('*.json')):
        report = json.loads(path.read_text())
        if report['condition'] not in ('diam', 'berjalan') or not report['completed'] or report['results'].get('exceptions', 0):
            continue
        frames = read_csv(report['raw_measurements']['timing_csv'])
        successful = [row for row in frames if not row['error'] and row['latency_ms']]
        elapsed = report['setup']['benchmark_seconds_measured']
        if not successful or elapsed <= 0:
            raise ValueError(f'Completed run has no valid measurements: {path}')
        if len(successful) != report['results']['frames_processed']:
            raise ValueError(f'Raw frame count disagrees with run metadata: {path}')
        values = stats([float(row['latency_ms']) for row in successful])
        summaries.append({'condition': report['condition'], 'run_id': report['run_id'],
                          'frames': len(successful), 'duration_seconds': elapsed,
                          'latency_mean_ms': values['mean'], 'latency_sd_ms': values['sd'],
                          'fps': len(successful)/elapsed})
        samples.extend(read_csv(report['raw_measurements']['resource_csv']))
        reports.append(report)
    if not summaries:
        raise ValueError('No completed stationary/walking experiments; no tables generated')
    resources = []
    for condition in ('diam', 'berjalan'):
        selected = [r for r in samples if r['condition'] == condition]
        row = {'condition': condition, 'samples': len(selected),
               'runs': sum(r['condition'] == condition for r in summaries)}
        for field in ('cpu_usage_percent', 'memory_used_mb', 'memory_usage_percent', 'detector_rss_mb'):
            for name, value in stats([float(r[field]) for r in selected]).items():
                row[f'{field}_{name}'] = value
        resources.append(row)
    write_table(root/'tables/table_3_1_latency.csv', summaries)
    write_table(root/'tables/table_3_2_resources_raw.csv', samples, markdown=False)
    write_table(root/'tables/table_3_3_resources_summary.csv', resources)
    write_table(root/'latency/runs_summary.csv', summaries)
    write_table(root/'resources/resources_all.csv', samples, markdown=False)
    write_table(root/'resources/resources_summary.csv', resources)
    metadata = {'ddof': 1, 'fps_definition': 'successful measured frames / measured wall duration',
                'latency_definition': 'image callback entry through postprocess completion, before publication',
                'memory_unit': 'MB = bytes / 1000000', 'cpu_main_metric': 'whole-system CPU percent',
                'resource_aggregation': 'all periodic samples pooled by condition',
                'missing_conditions': [r['condition'] for r in resources if not r['samples']],
                'completed_runs': len(summaries), 'system_ram_definition': 'psutil.virtual_memory.used; percent directly from psutil', 'model_identity': 'user-supplied QAT path; hashes not checked'}
    (root/'tables/metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    condition_note = ('Kondisi berjalan belum diukur; script disiapkan untuk operator.'
                      if 'berjalan' in metadata['missing_conditions'] else
                      'Kedua kondisi diam dan berjalan mempunyai run aktual; lihat jumlah run/sampel di tabel.')
    if all(r['samples'] for r in resources):
        deltas = []
        for field in ('cpu_usage_percent', 'memory_used_mb', 'memory_usage_percent', 'detector_rss_mb'):
            stationary, walking = (r[f'{field}_mean'] for r in resources)
            deltas.append({'metric': field, 'walking_minus_stationary': walking-stationary,
                           'relative_change_percent': 100*(walking-stationary)/stationary if stationary else None})
        write_table(root/'tables/condition_deltas.csv', deltas)
    protocol_path = root/'protocol.json'
    model_path = json.loads(protocol_path.read_text())['model_path'] if protocol_path.exists() else 'not recorded'
    sections = ['# Final OP3 QAT runtime report',
                'Model OpenVINO yang digunakan: `'+model_path+'` (XML dan BIN berpasangan).',
                'Model label: QAT INT8 OpenVINO, berdasarkan folder model yang diberikan pengguna. Identitas artefak tidak diverifikasi dengan hash.',
                'Hardware dan versi software: lihat environment_*.json. CPU onboard aktual NUC12; batch 1, input 640 × 640, CPU.',
                'Protocol dibekukan sebelum run di protocol.json. Tidak ada training, calibration, atau tuning.',
                '## Latency dan FPS', (root/'tables/table_3_1_latency.md').read_text(),
                '## CPU dan RAM', (root/'tables/table_3_3_resources_summary.md').read_text(),
                'CPU utama adalah CPU sistem. CPU proses dapat >100% karena multi-core. RAM utama adalah sistem dalam MB desimal; RSS proses disimpan terpisah.',
                'SD memakai ddof=1. Sampel resource dipool per kondisi; frame warmup dikecualikan. Kamera/ROS publishing aktif, debug image dan benchmark telemetry OFF.',
                '## Keterbatasan',
                condition_note+' Script tidak mengirim perintah walking. Dropped frame kamera tidak dapat ditentukan hanya dari callback yang diterima.',
                'Precision, Recall, F1 dan mAP tidak dihitung dari live camera. Laporan ini hanya mengukur runtime dan resource.',
                'Run dengan exception/thermal abort tidak dianggap berhasil oleh runner. Tabel hanya mengambil run completed; CSV raw dan metadata disimpan.',
                '## Reproduksi', 'Lihat command aktual dalam environment_*.json dan tiap run_metadata/*.json. Jalankan scripts/deployment/build_op3_thesis_tables.py --root '+str(root.resolve())]
    ancillary = root/'ancillary_report.md'
    if ancillary.exists():
        sections.append(ancillary.read_text())
    (root/'FINAL_OP3_QAT_REPORT.md').write_text('\n\n'.join(sections).rstrip()+'\n')
    return summaries, resources


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    build(parser.parse_args().root)
