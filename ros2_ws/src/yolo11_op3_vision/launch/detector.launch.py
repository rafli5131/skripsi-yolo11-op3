from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    names = (
        "model_path",
        "input_topic",
        "device",
        "imgsz",
        "confidence_threshold",
        "iou_threshold",
        "publish_debug_image",
        "debug_image_topic",
        "benchmark_seconds",
        "warmup_iterations",
        "resource_interval_seconds",
        "output_dir",
        "result_stem",
        "evidence_image_path",
    )
    extra = {"warmup_camera_frames": ("30", int), "publish_benchmark": ("false", bool),
             "benchmark_topic": ("/vision/benchmark", str), "experiment_condition": ("functional", str),
             "run_id": ("run_01", str), "final_experiment": ("false", bool), "thermal_limit_c": ("90.0", float)}
    names += tuple(extra)
    launch_arguments = names + ("python_prefix",)
    defaults = {
        "model_path": "",
        "input_topic": "/image_raw",
        "python_prefix": "",
        "device": "CPU",
        "imgsz": "640",
        "confidence_threshold": "0.25",
        "iou_threshold": "0.45",
        "publish_debug_image": "false",
        "debug_image_topic": "/vision/debug_image",
        "benchmark_seconds": "0.0",
        "warmup_iterations": "30",
        "resource_interval_seconds": "0.2",
        "output_dir": "",
        "result_stem": "ros2_benchmark",
        "evidence_image_path": "",
    }
    value_types = {
        "imgsz": int,
        "confidence_threshold": float,
        "iou_threshold": float,
        "publish_debug_image": bool,
        "benchmark_seconds": float,
        "warmup_iterations": int,
        "resource_interval_seconds": float,
    }
    defaults.update({k: v[0] for k, v in extra.items()})
    value_types.update({k: v[1] for k, v in extra.items()})
    return LaunchDescription(
        [
            *[
                DeclareLaunchArgument(name, default_value=defaults[name])
                for name in launch_arguments
            ],
            Node(
                package="yolo11_op3_vision",
                executable="detector",
                name="yolo11_op3_vision",
                output="screen",
                prefix=LaunchConfiguration("python_prefix"),
                parameters=[
                    {
                        name: ParameterValue(LaunchConfiguration(name), value_type=value_types[name])
                        if name in value_types
                        else LaunchConfiguration(name)
                        for name in names
                    }
                ],
            ),
        ]
    )
