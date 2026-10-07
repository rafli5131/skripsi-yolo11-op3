# Deployment Tools

Folder ini khusus untuk implementasi dan pengukuran pada ROBOTIS OP3.

- `audit_hardware_op3.py`: audit perangkat dan runtime robot.
- `audit_op3_models.py`: audit artefak model untuk deployment.
- `benchmark_op3.py`: benchmark OpenVINO standalone.
- `benchmark_op3_pytorch.py`: benchmark PyTorch pembanding.
- `benchmark_op3_ros2.py`: benchmark pipeline ROS 2.
- `capture_op3_evidence.py`: capture bukti visual/telemetry.

Pipeline penelitian utama tetap berada pada `scripts/step01_*.py` sampai
`scripts/step09_*.py`. Tool di folder ini dipakai setelah STEP 09.
