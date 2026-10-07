"""Connect thesis detections to the existing ROBOTIS soccer demo.

Observe mode creates no actuator publishers and never starts the demo.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Int32, String
from vision_msgs.msg import Detection2DArray
from op3_ball_detector_msgs.msg import CircleSetStamped


CONTROL_TOPICS = (
    '/ball_detector_node/circle_set', '/robotis/head_control/set_joint_states',
    '/robotis/head_control/set_joint_states_offset', '/robotis/walking/set_params',
    '/robotis/action/page_num', '/robotis/demo_command', '/thesis_soccer/game_state',
    '/robotis/walking/command', '/robotis/enable_ctrl_module',
)
BRIDGE_TOPICS = frozenset(('/ball_detector_node/circle_set', '/robotis/demo_command',
                          '/thesis_soccer/game_state', '/robotis/walking/command'))


def conflicting_publishers(node):
    """Ignore our own endpoints and the manager's built-in module selectors."""
    conflicts = []
    for topic in CONTROL_TOPICS:
        for endpoint in node.get_publishers_info_by_topic(topic):
            identity = (endpoint.node_namespace, endpoint.node_name)
            if topic in BRIDGE_TOPICS and identity == (node.get_namespace(), node.get_name()):
                continue
            if topic == '/robotis/enable_ctrl_module' and identity in (
                ('/', 'op3_base_module'), ('/', 'op3_tuning_module'),
            ):
                continue
            conflicts.append(f'{topic} <- {endpoint.node_namespace.rstrip("/")}/{endpoint.node_name}')
    return conflicts


def ball_candidate(message, width, height, confidence):
    """Return one ball: normalized center, pixel radius, confidence."""
    candidates = []
    if width <= 0 or height <= 0:
        return None
    for detection in message.detections:
        for result in detection.results:
            hypothesis = result.hypothesis
            if hypothesis.class_id.split(':')[0] != '0':
                continue
            box = detection.bbox
            values = (box.center.position.x, box.center.position.y,
                      box.size_x, box.size_y, hypothesis.score)
            if not all(math.isfinite(v) for v in values):
                continue
            x, y, w, h, score = values
            if score < confidence or w <= 0 or h <= 0 or not (0 <= x <= width and 0 <= y <= height):
                continue
            candidates.append((2*x/width-1, 2*y/height-1, min(w, h)/2, score))
    return max(candidates, key=lambda item: item[3]) if candidates else None


class SoccerBridge(Node):
    def __init__(self, args):
        super().__init__('thesis_soccer_bridge')
        self.args = args
        self.width = self.height = 0
        self.last_image = self.last_detection = self.last_ball = self.last_imu = 0.0
        self.started = False
        self.fault = None
        self.frames = self.balls = 0
        self.raw = (args.output / 'ball_adapter.csv').open('x', newline='')
        self.raw_writer = csv.writer(self.raw)
        self.raw_writer.writerow(('timestamp_ns', 'ball_found', 'normalized_x', 'normalized_y', 'radius_px', 'confidence'))
        self.circle_topic = '/thesis_soccer/preview/circle_set' if args.observe_only else '/ball_detector_node/circle_set'
        self.circle_pub = self.create_publisher(CircleSetStamped, self.circle_topic, 10)
        self.game_pub = self.stop_pub = self.demo_pub = None
        if not args.observe_only:
            self.game_pub = self.create_publisher(Int32, '/thesis_soccer/game_state', 10)
            self.stop_pub = self.create_publisher(String, '/robotis/walking/command', 10)
            self.demo_pub = self.create_publisher(String, '/robotis/demo_command', 10)
            self.create_subscription(Int32, '/thesis_soccer/game_state', self.game_request, 10)
        self.create_subscription(Image, '/thesis_soccer/image_raw', self.image, qos_profile_sensor_data)
        self.create_subscription(Detection2DArray, '/thesis_soccer/detections', self.detection, 10)
        self.create_subscription(Imu, '/robotis/open_cr/imu', self.imu, qos_profile_sensor_data)
        if getattr(args, 'show', False):
            import cv2
            from cv_bridge import CvBridge
            self.cv2, self.cv_bridge = cv2, CvBridge()
            self.create_subscription(Image, '/thesis_soccer/debug_image', self.debug_image, qos_profile_sensor_data)
        self.create_timer(.1, self.watchdog)

    def debug_image(self, msg):
        self.cv2.imshow('OP3 soccer - q/Esc stops this session', self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8'))
        if self.cv2.waitKey(1) & 0xff in (27, ord('q')):
            self.fault = 'operator closed visual window'

    def game_request(self, msg):
        if self.started and msg.data == 4:
            self.fault = 'operator requested soccer stop'

    def image(self, msg):
        self.width, self.height = msg.width, msg.height
        self.last_image = time.monotonic()

    def imu(self, msg):
        q = msg.orientation
        values = (q.x, q.y, q.z, q.w, msg.angular_velocity.x, msg.angular_velocity.y)
        if all(math.isfinite(v) for v in values) and .5 < q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w < 1.5:
            self.last_imu = time.monotonic()

    def detection(self, msg):
        now = time.monotonic()
        # Reject detections queued long after camera capture.
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        age = self.get_clock().now().nanoseconds / 1e9 - stamp
        if stamp <= 0 or age < -.2 or age > self.args.timeout:
            return
        self.last_detection = now
        self.frames += 1
        candidate = ball_candidate(msg, self.width, self.height, self.args.confidence)
        if now - self.last_image >= self.args.timeout:
            candidate = None
        result = CircleSetStamped()
        result.header = msg.header
        if candidate:
            x, y, radius, score = candidate
            result.circles = [Point(x=float(x), y=float(y), z=float(radius))]
            result.confidences = [float(score)]
            self.last_ball = now
            self.balls += 1
        self.circle_pub.publish(result)
        self.raw_writer.writerow((time.time_ns(), bool(candidate), *(candidate or ('', '', '', ''))))
        if self.frames % 30 == 0:
            self.raw.flush()

    def stop(self):
        if self.game_pub is not None:
            self.game_pub.publish(Int32(data=4))
            self.demo_pub.publish(String(data='stop'))
            self.stop_pub.publish(String(data='stop'))

    def watchdog(self):
        if not self.started:
            return
        now = time.monotonic()
        if now - min(self.last_image, self.last_detection, self.last_imu) > self.args.timeout:
            self.fault = 'camera, detector, or IMU stale'
        elif now - self.last_ball > self.args.ball_timeout:
            self.fault = 'ball lost; restart soccer explicitly to resume'
        if self.fault:
            self.stop()


def terminate(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--camera-device', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observe-only', action='store_true')
    parser.add_argument('--operator-confirmed', action='store_true')
    parser.add_argument('--seconds', type=float, default=0)
    parser.add_argument('--timeout', type=float, default=1.5)
    parser.add_argument('--ball-timeout', type=float, default=3)
    parser.add_argument('--confidence', type=float, default=.4)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    if not args.observe_only and not args.operator_confirmed:
        parser.error('Live soccer needs an operator beside the supported robot.')
    if args.show and not os.environ.get('DISPLAY'):
        parser.error('--show requires working X11 DISPLAY')
    if args.timeout <= 0 or args.ball_timeout <= 0 or args.seconds < 0 or not 0 <= args.confidence <= 1:
        parser.error('Invalid timeout, duration, or confidence')
    if not args.model.is_file() or not Path(args.camera_device).exists():
        parser.error('Model/camera path does not exist')
    args.output.mkdir(parents=True, exist_ok=False)
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = SoccerBridge(args)
    processes, logs = [], []
    began = time.monotonic()
    reason = 'operator stopped'
    demo = None
    def launch(name, command):
        log = (args.output / f'{name}.log').open('x')
        logs.append(log)
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process
    def spin():
        rclpy.spin_once(node, timeout_sec=.1)
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f'Child exited ({process.returncode}); inspect session logs')
    try:
        # Discover existing owners before spawning anything; no robot controller startup here.
        for _ in range(15):
            rclpy.spin_once(node, timeout_sec=.1)
        if not args.observe_only:
            names = node.get_node_names()
            if 'op3_manager' not in names:
                raise RuntimeError('Start manager and init pose from operator menu first')
            if any('motion_gateway' in name or name in ('demo_node', 'thesis_soccer_demo') for name in names):
                raise RuntimeError('Another soccer/motion owner is active')
            conflicts = conflicting_publishers(node)
            if conflicts:
                raise RuntimeError('An existing publisher owns soccer/head/walking/action control: '
                                   + '; '.join(conflicts))
        launch('camera', ['ros2', 'run', 'usb_cam', 'usb_cam_node_exe', '--ros-args',
               '-r', '__node:=thesis_soccer_camera', '-r', 'image_raw:=/thesis_soccer/image_raw',
               '-p', f'video_device:={args.camera_device}', '-p', 'image_width:=640',
               '-p', 'image_height:=480', '-p', 'framerate:=30.0'])
        launch('detector', [sys.executable, '-m', 'yolo11_op3_vision.detector', '--ros-args',
               '-r', '__node:=thesis_soccer_detector', '-r', '/vision/detections:=/thesis_soccer/detections',
               '-p', f'model_path:={args.model.resolve()}', '-p', 'input_topic:=/thesis_soccer/image_raw',
               '-p', 'publish_debug_image:=true', '-p', 'debug_image_topic:=/thesis_soccer/debug_image',
               '-p', f'evidence_image_path:={args.output.resolve() / "detection.png"}',
               '-p', f'confidence_threshold:={args.confidence}'])
        deadline = time.monotonic() + 40
        while node.frames == 0:
            spin()
            if time.monotonic() > deadline:
                raise RuntimeError('No fresh detections within startup deadline')
        if not args.observe_only:
            if time.monotonic() - node.last_ball > args.timeout or time.monotonic() - node.last_imu > args.timeout:
                raise RuntimeError('Show a ball to the camera; fresh ball and IMU required before soccer startup')
            print('Starting existing soccer: walking-ready pose, head tracking, following, kicks and fall recovery.', flush=True)
            demo = launch('soccer_demo', ['ros2', 'run', 'op3_demo', 'op_demo_node', '--ros-args',
                          '-r', '__node:=thesis_soccer_demo', '-r', 'game_controller:=/thesis_soccer/game_state'])
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                spin()
            if not node.count_subscribers('/thesis_soccer/game_state'):
                raise RuntimeError('Soccer game-state subscriber missing')
            if time.monotonic()-node.last_ball > args.timeout or time.monotonic()-node.last_imu > args.timeout:
                raise RuntimeError('Ball or IMU stale after walking-ready pose')
            node.started = True
            for _ in range(3):
                node.game_pub.publish(Int32(data=3))
                spin()
        print('Observe only: no motor commands.' if args.observe_only else 'Soccer active. Ctrl+C to stop.', flush=True)
        active_began = time.monotonic()
        while not args.seconds or time.monotonic() - active_began < args.seconds:
            spin()
            if node.fault:
                raise RuntimeError(node.fault)
        reason = 'requested duration finished'
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        reason = str(exc)
        raise
    finally:
        # Only send stop if this session actually started the actuator-owning demo.
        if demo is not None:
            for _ in range(3):
                node.stop()
                time.sleep(.1)
            terminate(demo)
            node.stop()
            time.sleep(.2)
        for process in reversed(processes):
            terminate(process)
        node.raw.close()
        (args.output / 'session.json').write_text(json.dumps({
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'observe_only': args.observe_only, 'model_path': str(args.model.resolve()),
            'camera_device': args.camera_device, 'elapsed_seconds': time.monotonic()-began,
            'detection_frames': node.frames, 'ball_frames': node.balls,
            'soccer_started': node.started, 'stop_reason': reason,
            'circle_topic': node.circle_topic, 'actuator_demo_started': demo is not None,
            'command': sys.argv,
        }, indent=2)+'\n')
        for log in logs:
            log.close()
        if args.show:
            node.cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
