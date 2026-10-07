"""Adapter/control gates; these tests never publish actuator commands."""
from argparse import Namespace
from pathlib import Path
import tempfile
import time
import unittest

import rclpy
from std_msgs.msg import Int32
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from yolo11_op3_vision.soccer import SoccerBridge, ball_candidate


def detection(class_id='0:ball', score=.9, x=480.):
    msg = Detection2DArray()
    item = Detection2D()
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id, result.hypothesis.score = class_id, score
    item.results = [result]
    item.bbox.center.position.x, item.bbox.center.position.y = x, 120.
    item.bbox.size_x, item.bbox.size_y = 40., 60.
    msg.detections = [item]
    return msg


class BallAdapterTests(unittest.TestCase):
    def test_geometry_matches_existing_soccer(self):
        self.assertEqual(ball_candidate(detection(), 640, 480, .4), (.5, -.5, 20., .9))

    def test_reject_non_ball_low_confidence_invalid_geometry(self):
        for msg in (detection('2:robot'), detection(score=.1), detection(x=float('nan')), detection(x=700.)):
            self.assertIsNone(ball_candidate(msg, 640, 480, .4))
        self.assertIsNone(ball_candidate(detection(), 0, 480, .4))

    def test_choose_one_highest_confidence_ball(self):
        msg = detection(score=.5)
        msg.detections += detection(score=.95, x=320.).detections
        self.assertEqual(ball_candidate(msg, 640, 480, .4), (0., -.5, 20., .95))

    def test_observe_has_no_actuator_publishers_and_watchdog_latches(self):
        with tempfile.TemporaryDirectory() as directory:
            rclpy.init()
            node = SoccerBridge(Namespace(observe_only=True, output=Path(directory),
                                timeout=1.5, ball_timeout=3., confidence=.4, show=False))
            try:
                self.assertIsNone(node.game_pub)
                self.assertIsNone(node.demo_pub)
                self.assertIsNone(node.stop_pub)
                self.assertEqual(node.circle_topic, '/thesis_soccer/preview/circle_set')
                node.started = True
                node.last_image = node.last_detection = node.last_imu = time.monotonic()
                node.last_ball = time.monotonic()-4
                node.watchdog()
                self.assertIn('ball lost', node.fault)
                node.last_ball = time.monotonic()
                node.watchdog()
                self.assertIn('ball lost', node.fault)
                node.fault = None
                node.last_imu = time.monotonic()-2
                node.watchdog()
                self.assertIn('IMU stale', node.fault)
                node.game_request(Int32(data=4))
                self.assertEqual(node.fault, 'operator requested soccer stop')
            finally:
                node.raw.close()
                node.destroy_node()
                rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
