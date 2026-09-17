import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class ArmTracker:
    def __init__(self):
        model_path = Path(__file__).with_name("pose_landmarker_lite.task")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        )
        self.last_timestamp_ms = 0
        self.previous_x = None
        self.previous_y = None

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Video timestamps must always increase, even for very fast frames.
        timestamp_ms = max(
            time.monotonic_ns() // 1_000_000,
            self.last_timestamp_ms + 1,
        )
        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        left_shoulder = landmarks[mp.tasks.vision.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.tasks.vision.PoseLandmark.RIGHT_SHOULDER]
        wrist = landmarks[mp.tasks.vision.PoseLandmark.RIGHT_WRIST]

        if min(
            left_shoulder.visibility,
            right_shoulder.visibility,
            wrist.visibility,
        ) < 0.5:
            return None

        # Shoulder width makes the hand offset independent of camera distance.
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        scale = max(shoulder_width * 2.0, 0.1)
        current_x = np.clip((wrist.x - right_shoulder.x) / scale, -1.0, 1.0)
        current_y = np.clip((right_shoulder.y - wrist.y) / scale, -1.0, 1.0)

        if self.previous_x is None:
            self.previous_x = current_x
            self.previous_y = current_y
        else:
            self.previous_x = 0.8 * self.previous_x + 0.2 * current_x
            self.previous_y = 0.8 * self.previous_y + 0.2 * current_y

        return float(self.previous_x), float(self.previous_y)

    def close(self):
        self.landmarker.close()
