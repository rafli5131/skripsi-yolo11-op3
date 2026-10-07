#!/usr/bin/env python3
"""Run stationary/operator-controlled camera experiments without commanding motion."""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['diam', 'berjalan', 'functional', 'stability'], required=True)
    parser.add_argument('--operator-confirmed', action='store_true')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--run-start', type=int, default=1)
    parser.add_argument('--camera-device', required=True)
    parser.add_argument('--input-topic', default='/thesis/image_raw')
    parser.add_argument('--model', type=Path, default=ROOT/'artifacts/openvino/qat_int8/model.xml')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/hardware_op3/final_qat')
    args = parser.parse_args()
    functional = args.condition == 'functional'
    if args.condition in ('diam', 'berjalan') and not args.operator_confirmed:
        parser.error('Confirm actual robot condition with --operator-confirmed; this script never moves the robot')
    if args.runs < 1 or args.seconds <= 0 or args.run_start < 1:
        parser.error('Runs, duration and start index must be positive')
    if args.condition == 'stability' and args.seconds < 600:
        parser.error('Stability measurements require at least 600 seconds')
    camera_path = Path(args.camera_device).resolve()
    if not camera_path.exists():
        parser.error('Audited camera device does not exist')
    if not os.environ.get('ROS_DISTRO'):
        parser.error('Source /opt/ros/jazzy/setup.bash and the ROS workspace first')
    subprocess.run(['ros2', 'pkg', 'prefix', 'usb_cam'], check=True, stdout=subprocess.DEVNULL)
    args.output.mkdir(parents=True, exist_ok=True)
    for number in range(args.run_start, args.run_start+args.runs):
        key = f'{args.condition}_run_{number:02d}'
        paths = (args.output/'latency'/f'{key}_frames.csv', args.output/'resources'/f'{key}.csv',
                 args.output/'run_metadata'/f'{key}.json') if not functional else (
                 args.output/f'{key}.csv', args.output/f'resources_{key}.csv', args.output/f'{key}.json')
        for path in paths:
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite measured run: {path}')
    protocol = {'model_path': str(args.model.resolve()), 'device': 'CPU', 'imgsz': 640,
                'batch': 1, 'seconds': args.seconds, 'runs': args.runs,
                'warmup_inferences': 30, 'warmup_camera_frames': 30,
                'resource_interval_seconds': .2, 'confidence_threshold': .25,
                'iou_threshold': .45, 'camera_device': str(camera_path),
                'capture_resolution': [640, 480], 'capture_framerate': 30,
                'publish_debug_image': functional, 'publish_benchmark': functional}
    protocol_path = args.output/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError('Protocol differs from frozen experiment; choose a new output directory')
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    import openvino as ov
    import psutil
    environment = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                   'condition': args.condition, 'operator_confirmed': args.operator_confirmed,
                   'camera_name': Path('/sys/class/video4linux/'+camera_path.name+'/name').read_text().strip(),
                   'hostname': platform.node(), 'python_version': platform.python_version(),
                   'openvino_version': ov.__version__, 'available_devices': ov.Core().available_devices,
                   'ros_distro': os.environ.get('ROS_DISTRO'),
                   'cpu': subprocess.check_output(['lscpu'], text=True),
                   'ram_total_bytes': psutil.virtual_memory().total,
                   'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   'git_worktree': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                   'command': [sys.executable, *sys.argv],
                   'artifact_identity_verification': 'not performed; user-supplied QAT folder label'}
    key = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (args.output/f'environment_{key}.json').write_text(json.dumps(environment, indent=2)+'\n')
    camera = node = None
    env = os.environ.copy()
    source = ROOT/'ros2_ws/src/yolo11_op3_vision'
    env['PYTHONPATH'] = str(source)+os.pathsep+env.get('PYTHONPATH', '')
    logs = args.output/'logs'
    logs.mkdir(exist_ok=True)
    try:
        with (logs/f'camera_{key}.log').open('x') as log:
            camera = subprocess.Popen(['ros2', 'run', 'usb_cam', 'usb_cam_node_exe', '--ros-args',
                '-p', f'video_device:={camera_path}', '-p', 'image_width:=640', '-p', 'image_height:=480',
                '-p', 'framerate:=30.0', '-r', f'image_raw:={args.input_topic}'],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for number in range(args.run_start, args.run_start+args.runs):
                run_id = f'run_{number:02d}'
                params = {'model_path': str(args.model.resolve()), 'input_topic': args.input_topic,
                          'output_dir': str(args.output.resolve()), 'result_stem': f'{args.condition}_{run_id}',
                          'experiment_condition': args.condition, 'run_id': run_id,
                          'benchmark_seconds': args.seconds, 'final_experiment': not functional,
                          'warmup_camera_frames': 30, 'publish_benchmark': functional,
                          'publish_debug_image': functional}
                if functional:
                    params['evidence_image_path'] = str(args.output.resolve()/f'{args.condition}_{run_id}_detection.png')
                command = [sys.executable, '-m', 'yolo11_op3_vision.detector', '--ros-args']
                for name, value in params.items():
                    command += ['-p', f'{name}:={str(value).lower() if isinstance(value, bool) else value}']
                if camera.poll() is not None:
                    raise RuntimeError('Camera process exited; inspect camera log')
                with (logs/f'{args.condition}_{run_id}_{key}.log').open('x') as node_log:
                    print(f'Starting {args.condition}/{run_id}: {args.seconds}s', flush=True)
                    node = subprocess.Popen(command, env=env, stdout=node_log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    import time
                    deadline = time.monotonic()+args.seconds+120
                    while node.poll() is None:
                        if camera.poll() is not None:
                            raise RuntimeError('Camera process exited; inspect camera log')
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Camera/node did not finish within measurement watchdog')
                        time.sleep(.25)
                    code = node.returncode
                report_path = (args.output/f'{args.condition}_{run_id}.json') if functional else (args.output/'run_metadata'/f'{args.condition}_{run_id}.json')
                report = json.loads(report_path.read_text())
                if code or not report['completed'] or report['results']['frames_processed'] < 1 or report['results']['exceptions'] or report['results']['resource_sample_count'] < 1:
                    raise RuntimeError(f'Run not accepted: {report_path}; exit={code}')
                print(json.dumps({'run': run_id, 'condition': args.condition, 'results': report['results']}), flush=True)
    finally:
        stop(node)
        stop(camera)


if __name__ == '__main__':
    main()
