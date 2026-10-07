"""Streaming experiment records; byte and MB units stay explicit."""
from __future__ import annotations

import csv
import re
from pathlib import Path

FRAME_FIELDS = (
    'run_id', 'condition', 'frame_index', 'timestamp', 'preprocess_ms',
    'inference_ms', 'postprocess_ms', 'latency_ms', 'detection_count', 'error',
    'image_conversion_ms', 'callback_end_to_end_ms', 'camera_header_stamp_ns',
    'camera_delivery_ms',
)
RESOURCE_FIELDS = (
    'sample_no', 'run_id', 'condition', 'timestamp', 'elapsed_seconds',
    'cpu_usage_percent', 'memory_used_mb', 'memory_usage_percent',
    'detector_process_cpu_percent', 'detector_rss_mb', 'system_cpu_percent',
    'process_cpu_percent', 'rss_bytes', 'vms_bytes', 'system_memory_used_bytes',
    'system_memory_available_bytes', 'per_core_cpu_percent', 'temperature_c',
)


def temperatures_c():
    values = []
    for path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            value = float(path.read_text().strip()) / 1000
            if 0 < value < 150:
                values.append(value)
        except (OSError, ValueError):
            pass
    return values


class RunRecorder:
    def __init__(self, root: Path, stem: str, run_id: str, condition: str, final_layout: bool):
        for value in (stem, run_id):
            if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
                raise ValueError('Run IDs/stems must use letters, numbers, underscores, or hyphens')
        if final_layout:
            key = f'{condition}_{run_id}'
            self.frame_path = root / 'latency' / f'{key}_frames.csv'
            self.resource_path = root / 'resources' / f'{key}.csv'
            self.report_path = root / 'run_metadata' / f'{key}.json'
        else:
            self.frame_path = root / f'{stem}.csv'
            self.resource_path = root / f'resources_{stem}.csv'
            self.report_path = root / f'{stem}.json'
        for path in (self.frame_path, self.resource_path, self.report_path):
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite experiment data: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_stream = self.frame_path.open('x', newline='', encoding='utf-8')
        self.resource_stream = self.resource_path.open('x', newline='', encoding='utf-8')
        self.frame_writer = csv.DictWriter(self.frame_stream, fieldnames=FRAME_FIELDS, extrasaction='raise')
        self.resource_writer = csv.DictWriter(self.resource_stream, fieldnames=RESOURCE_FIELDS, extrasaction='raise')
        self.frame_writer.writeheader()
        self.resource_writer.writeheader()

    def frame(self, row):
        self.frame_writer.writerow(row)
        self.frame_stream.flush()

    def resource(self, row):
        self.resource_writer.writerow(row)
        self.resource_stream.flush()

    def close(self):
        self.frame_stream.close()
        self.resource_stream.close()

    def rows(self):
        with self.frame_path.open(newline='') as stream:
            frames = list(csv.DictReader(stream))
        with self.resource_path.open(newline='') as stream:
            resources = list(csv.DictReader(stream))
        return frames, resources
