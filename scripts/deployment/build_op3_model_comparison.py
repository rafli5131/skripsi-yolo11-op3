#!/usr/bin/env python3
"""Compare fresh four-model CPU runs; keep dataset accuracy and ROS runs separate."""
import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = [('fp32_pytorch', 'FP32 PyTorch (tanpa OpenVINO)'),
          ('fp32', 'FP32 OpenVINO'), ('ptq_int8', 'PTQ INT8 OpenVINO'),
          ('qat_int8', 'QAT INT8 OpenVINO')]


def plot_camera(root, rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    camera = [r for r in rows if r['mode'] == 'camera']
    labels = ['FP32 PyTorch\ntanpa OpenVINO', 'FP32\nOpenVINO', 'PTQ INT8\nOpenVINO', 'QAT INT8\nOpenVINO']
    import sys
    sys.path.insert(0, str(REPO))
    from scripts.reporting.figure_style import comparison, configure
    configure()
    fig, axes = plt.subplots(4, 2, figsize=(15, 17))
    for pair, field, title in zip(axes,
            ['inference_mean_ms', 'effective_fps', 'system_cpu_mean_percent', 'rss_peak_mb'],
            ['Inference mean (ms)', 'Effective real-camera FPS', 'System CPU mean (%)', 'Detector RSS peak (MB decimal)']):
        values = [float(r[field]) for r in camera]
        comparison(*pair, labels, {'Measured': values}, title, decimals=3)
    fig.suptitle('OP3 / NUC12: measured standalone C920 comparison\nCPU, batch 1, model 640×640, warmup 30, camera 60s/model, visualization OFF')
    fig.subplots_adjust(top=.92, bottom=.08, left=.09, right=.97, wspace=.50, hspace=.90)
    fig.text(.5, .015, 'Kamera C920; durasi 60 detik per model. Skala panel kanan diperbesar.', ha='center')
    output = root/'evidence/four_model_camera_comparison.png'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def write_table(path, rows):
    fields = list(rows[0])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    def cell(value):
        return f'{value:.4f}' if isinstance(value, float) else str(value)
    text = ['| '+' | '.join(fields)+' |', '| '+' | '.join(['---']*len(fields))+' |']
    text += ['| '+' | '.join(cell(row[k]) for k in fields)+' |' for row in rows]
    path.with_suffix('.md').write_text('\n'.join(text)+'\n')


def build(root):
    rows, provenance = [], []
    for mode in ('synthetic', 'camera'):
        for key, label in MODELS:
            filename = ('benchmark_fp32_pytorch' if mode == 'synthetic' else 'camera_benchmark_fp32_pytorch') if key == 'fp32_pytorch' else f'{mode}_{key}'
            path = root/f'{filename}.json'
            record = json.loads(path.read_text())
            setup = record['experimental_setup']
            assert setup['device'].upper() == 'CPU' and setup['batch'] == 1
            assert setup['input_resolution'] == [640, 640] and setup['warmups'] >= 30
            assert setup['mode'] == mode and record['official_minimums_met']
            assert not setup.get('visualization_enabled', setup.get('visualization_enabled_during_measurement', False))
            if key == 'fp32_pytorch':
                assert setup['openvino_used'] is False and setup['rect'] is False
            if mode == 'synthetic':
                assert setup['iterations_processed'] == 300
            else:
                assert setup['duration_requested_seconds'] == 60
            for name in ('raw_measurements_csv', 'raw_resource_samples_csv'):
                assert (REPO/record[name]).is_file(), f'Missing actual raw data: {record[name]}'
            resources = record['resource_summary']
            latency = record['latency_ms']
            loop = latency['end_to_end_wall_ms'] if key == 'fp32_pytorch' else latency['loop_wall_ms']
            row = {'mode': mode, 'model': label,
                   'inference_mean_ms': latency['inference_ms']['mean'],
                   'inference_p95_ms': latency['inference_ms']['p95'],
                   'pipeline_mean_ms': loop['mean'],
                   'effective_fps': record['effective_end_to_end_fps'],
                   'system_cpu_mean_percent': resources['system_cpu_percent']['mean'],
                   'process_cpu_mean_percent': resources['process_cpu_percent']['mean'],
                   'system_memory_mean_mb': resources['system_memory_used_bytes']['mean']/1e6,
                   'rss_mean_mb': resources['rss_bytes']['mean']/1e6,
                   'rss_peak_mb': resources['rss_bytes']['max']/1e6}
            rows.append(row)
            provenance.append({'mode': mode, 'model': label, 'report': str(path.resolve()),
                               'model_path': record['model'].get('path', record['model'].get('xml_path')),
                               'git_commit': record['git_commit'], 'command': record['command'],
                               'raw_timing_csv': record['raw_measurements_csv'],
                               'raw_resources_csv': record['raw_resource_samples_csv']})
    write_table(root/'comparison_all.csv', rows)
    thesis = [r for r in rows if r['model'] != 'PTQ INT8 OpenVINO']
    write_table(root/'comparison_thesis_fp32_qat.csv', thesis)
    for mode in ('synthetic', 'camera'):
        write_table(root/f'comparison_{mode}.csv', [r for r in rows if r['mode'] == mode])
    accuracy = []
    for key, label in [('fp32', 'FP32 native PyTorch checkpoint'), ('qat', 'QAT native PyTorch checkpoint')]:
        source = REPO/f'experiments/final_test/{key}_test_metrics.json'
        metrics = json.loads(source.read_text())['overall']
        accuracy.append({'model': label, 'split': 'TEST',
                         **{k: metrics[k] for k in ('precision', 'recall', 'F1', 'mAP50', 'mAP50-95')}})
    write_table(root/'accuracy_frozen_fp32_qat.csv', accuracy)
    (root/'comparison.json').write_text(json.dumps({'runtime': rows, 'thesis_runtime': thesis,
        'accuracy_native_pytorch_only': accuracy, 'provenance': provenance,
        'accuracy_note': 'Frozen native checkpoint evaluation; not measured IR/live-camera accuracy'}, indent=2)+'\n')
    sections = ['# Perbandingan empat model pada CPU OP3',
        'Pengukuran baru menggunakan FP32 PyTorch tanpa OpenVINO, FP32 OpenVINO, PTQ INT8 OpenVINO, dan QAT INT8 OpenVINO. Semua IR diambil dari artifacts/openvino/{fp32,ptq_int8,qat_int8}/model.xml; FP32 native dari artifacts/checkpoints/fp32/best.pt. Label kuantisasi mengikuti artefak pengguna, tanpa pemeriksaan hash.',
        '## Protokol', 'CPU, batch 1, model input 640×640, 30 warmup, synthetic 300 iteration dengan seed 42, kamera 60 detik per model. Confidence 0.25, IoU 0.45, sampling resource 0.2 detik, visualization OFF. Native PyTorch memakai rect=False agar kamera 640×480 tetap di-letterbox ke 640×640. Model loading/compile dikeluarkan dari waktu inference.',
        '## Standalone synthetic', (root/'comparison_synthetic.md').read_text(),
        '## Standalone kamera', (root/'comparison_camera.md').read_text(),
        '## Tabel untuk skripsi: FP32 dan QAT', (root/'comparison_thesis_fp32_qat.md').read_text(),
        'PTQ tersedia sebagai pembanding tambahan dalam comparison_all.csv; tidak termasuk comparison_thesis_fp32_qat.csv.',
        '## Accuracy dataset yang dibekukan', (root/'accuracy_frozen_fp32_qat.md').read_text(),
        'Accuracy di atas milik checkpoint native PyTorch pada TEST, bukan hasil kamera atau pengukuran ulang accuracy IR. Tidak ada training/calibration/tuning/evaluasi ulang TEST.',
        '## Batas perbandingan', 'Pengujian runtime dilakukan berurutan pada host yang sama. Kamera live tidak menjamin frame/scene identik antar model. OpenVINO memakai preprocessing/postprocess custom existing; native memakai Ultralytics. Timing inference native berasal dari profiler Ultralytics, OpenVINO memakai perf_counter_ns; pipeline wall time/FPS lebih mewakili biaya keseluruhan implementasi. Jumlah thread mengikuti default masing-masing runtime dan dicatat pada JSON jika tersedia. CPU sistem dan CPU proses terpisah; CPU proses dapat >100%. RAM MB = bytes/1000000.',
        'Hasil ROS QAT/stability tersimpan terpisah dalam final_qat/ dan stability_qat/. Tabel standalone ini tidak disebut sebagai perbandingan ROS empat model.',
        '## Reproduksi', 'bash scripts/run_op3_model_comparison.sh <direktori-output-baru>\n\npython3 scripts/deployment/build_op3_model_comparison.py --root '+str(root.resolve()),
        'Raw timing/resource CSV, timestamp, commit, hostname, model path, versi runtime dan command aktual ada pada setiap JSON benchmark. Protocol dibekukan di protocol.json sebelum run.']
    if plot_camera(root, rows):
        sections.append('Grafik dari hasil actual: [four_model_camera_comparison.png](evidence/four_model_camera_comparison.png).')
    (root/'MODEL_COMPARISON_REPORT.md').write_text('\n\n'.join(sections)+'\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    build(parser.parse_args().root)
