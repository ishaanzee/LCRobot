import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from vision.gestures import GestureDetector
from vision.gesture_process import GestureProcess


class GestureTests(unittest.TestCase):
    def test_recognizer_classes_control_light_without_body_landmarks(self):
        detector = GestureDetector.__new__(GestureDetector)
        detector.recognizer = Mock()
        detector.last_timestamp_ms = 0
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        for label, score, expected in (
            ("Open_Palm", 0.9, "on"), ("Closed_Fist", 0.9, "off"),
            ("Closed_Fist", 0.3, None), ("Thumb_Up", 0.9, None),
        ):
            detector.recognizer.recognize_for_video.return_value = SimpleNamespace(
                gestures=[[SimpleNamespace(category_name=label, score=score)]]
            )
            self.assertEqual(detector.detect(frame), expected)
        detector.recognizer.recognize_for_video.return_value.gestures = []
        self.assertIsNone(detector.detect(frame))

    def worker(self):
        with patch.object(GestureProcess, "start"):
            worker = GestureProcess()
        worker.process = Mock()
        worker.process.is_alive.return_value = True
        worker.connection = Mock()
        worker.connection.poll.return_value = False
        worker.started_at = 9
        worker.sent_at = 9.8
        return worker

    @patch("vision.gesture_process.time.monotonic", return_value=10)
    def test_waits_for_result_before_sending_another_frame(self, clock):
        worker = self.worker()
        self.assertIsNone(worker.detect("new frame"))
        worker.connection.send.assert_not_called()
        worker.connection.poll.return_value = True
        worker.connection.recv.return_value = ("result", "off")
        self.assertEqual(worker.detect("new frame"), "off")
        worker.connection.send.assert_called_once_with("new frame")

    @patch("vision.gesture_process.time.monotonic", return_value=10)
    def test_stale_results_cannot_change_light(self, clock):
        worker = self.worker()
        worker.sent_at = 8
        worker.connection.poll.return_value = True
        worker.connection.recv.return_value = ("result", "on")
        self.assertIsNone(worker.detect("current frame"))

    @patch("vision.gesture_process.time.monotonic", return_value=10)
    def test_native_worker_exit_is_contained_and_restarted(self, clock):
        worker = self.worker()
        worker.process.is_alive.return_value = False
        worker.process.exitcode = -5  # SIGTRAP, as in the reported crash.
        self.assertIsNone(worker.detect("frame"))
        self.assertIsNone(worker.process)
        self.assertEqual(worker.status, "restarting")
        with patch.object(worker, "start") as start:
            worker.detect("frame")
            start.assert_not_called()
            clock.return_value = 11
            worker.detect("frame")
            start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
