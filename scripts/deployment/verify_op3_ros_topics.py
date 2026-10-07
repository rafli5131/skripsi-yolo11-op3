#!/usr/bin/env python3
"""Observe actual thesis ROS publications during a separate functional run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seconds', type=float, default=20)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit('Refusing to overwrite functional topic evidence')
rclpy.init()
node = rclpy.create_node('thesis_functional_topic_verification')
counts = {'detections': 0, 'debug_image': 0, 'benchmark': 0}
last = {}

def detections(message):
    counts['detections'] += 1
    last['detections'] = [{'class_id': h.hypothesis.class_id, 'confidence': h.hypothesis.score}
                          for d in message.detections for h in d.results]

def benchmark(message):
    counts['benchmark'] += 1
    last['benchmark'] = json.loads(message.data)

def image(message):
    counts['debug_image'] += 1
    last['image_shape'] = [message.height, message.width]

subscriptions = [node.create_subscription(Detection2DArray, '/vision/detections', detections, 10),
                 node.create_subscription(String, '/vision/benchmark', benchmark, 100),
                 node.create_subscription(Image, '/vision/debug_image', image, qos_profile_sensor_data)]
started = time.monotonic()
try:
    while time.monotonic()-started < args.seconds:
        rclpy.spin_once(node, timeout_sec=.1)
finally:
    node.destroy_node()
    rclpy.shutdown()
args.output.parent.mkdir(parents=True, exist_ok=True)
report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'seconds_observed': time.monotonic()-started,
          'topic_message_counts': counts, 'last_received': last,
          'passed': all(counts.values()), 'purpose': 'functional verification; excluded from official benchmark'}
args.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(0 if report['passed'] else 1)
