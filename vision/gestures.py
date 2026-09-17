import time
from pathlib import Path

import cv2
import mediapipe as mp


class GestureDetector:
    def __init__(self):
        model_path = Path(__file__).with_name("gesture_recognizer.task")
        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(
            options
        )
        self.last_timestamp_ms = 0

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Video timestamps must always increase, even for very fast frames.
        timestamp_ms = max(
            time.monotonic_ns() // 1_000_000,
            self.last_timestamp_ms + 1,
        )
        self.last_timestamp_ms = timestamp_ms

        result = self.recognizer.recognize_for_video(image, timestamp_ms)
        if not result.gestures or not result.gestures[0]:
            return None

        gesture = result.gestures[0][0]
        if gesture.score < 0.6:
            return None
        if gesture.category_name == "Open_Palm":
            return "on"
        if gesture.category_name == "Closed_Fist":
            return "off"
        return None

    def close(self):
        self.recognizer.close()
