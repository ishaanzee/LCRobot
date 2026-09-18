import math
import time
from pathlib import Path

import cv2
import mediapipe as mp


class ArmTracker:
    """Measure shoulder rotation, upper-arm elevation, and elbow bend in 3D."""

    def __init__(self, arm="right"):
        if arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        self.arm = arm
        self.previous_angles = None
        self.last_seen = None
        self.arm_points = None
        self.smoothing_seconds = 0.08
        self.reset_after_seconds = 0.5
        model_path = Path(__file__).with_name("pose_landmarker_lite.task")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0

    def switch_arm(self):
        self.arm = "left" if self.arm == "right" else "right"
        self.previous_angles = None
        self.last_seen = None
        self.arm_points = None

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        now = time.monotonic()
        # Video timestamps must increase, even for very fast frames.
        timestamp_ms = max(int(now * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        self.arm_points = None
        if not result.pose_landmarks or not result.pose_world_landmarks:
            return None
        height, width = frame.shape[:2]
        return self.angles_from_landmarks(
            result.pose_landmarks[0], result.pose_world_landmarks[0],
            width, height, now,
        )

    def angles_from_landmarks(self, image_points, world_points, width, height, now):
        """Use image points for visibility/drawing and 3D points for angles."""
        self.arm_points = None
        pose = mp.tasks.vision.PoseLandmark
        shoulder_id = getattr(pose, f"{self.arm.upper()}_SHOULDER")
        elbow_id = getattr(pose, f"{self.arm.upper()}_ELBOW")
        wrist_id = getattr(pose, f"{self.arm.upper()}_WRIST")
        selected_points = [image_points[i] for i in (shoulder_id, elbow_id, wrist_id)]
        if any(
            not math.isfinite(point.visibility) or point.visibility < 0.5
            or not (0 <= point.x <= 1 and 0 <= point.y <= 1)
            for point in selected_points
        ):
            return None

        shoulder, elbow, wrist = [
            world_points[i] for i in (shoulder_id, elbow_id, wrist_id)
        ]
        if not all(
            math.isfinite(value)
            for point in (shoulder, elbow, wrist)
            for value in (point.x, point.y, point.z)
        ):
            return None

        # MediaPipe coordinates: x is image-right, y is down, z is away
        # from the camera. Use the upper arm, not the wrist, for the shoulder.
        upper_x = elbow.x - shoulder.x
        upper_y = elbow.y - shoulder.y
        upper_z = elbow.z - shoulder.z
        forearm_x = wrist.x - elbow.x
        forearm_y = wrist.y - elbow.y
        forearm_z = wrist.z - elbow.z
        upper_length = math.sqrt(upper_x**2 + upper_y**2 + upper_z**2)
        forearm_length = math.sqrt(forearm_x**2 + forearm_y**2 + forearm_z**2)
        if min(upper_length, forearm_length) < 0.05:
            return None  # 3D landmarks less than 5 cm apart are unreliable.

        recent_pose = (
            self.last_seen is not None and now - self.last_seen <= self.reset_after_seconds
        )
        previous_yaw = self.previous_angles[0] if recent_pose else 0.0
        horizontal_length = math.hypot(upper_x, upper_z)
        if horizontal_length < 0.15 * upper_length:
            # A nearly vertical arm has no reliable compass direction.
            yaw = previous_yaw
        else:
            # Toward the camera = lamp forward; image-left = lamp left.
            yaw = math.atan2(-upper_x, -upper_z)
            # Choose the nearest equivalent angle across the +/-pi boundary.
            # Then clamp to the base's soft limits so it cannot spin through
            # the front when noisy estimates alternate behind the person.
            difference = math.atan2(
                math.sin(yaw - previous_yaw), math.cos(yaw - previous_yaw)
            )
            yaw = max(-2.45, min(2.45, previous_yaw + difference))
        elevation = math.atan2(-upper_y, horizontal_length)

        # Matching upper-arm and forearm directions mean a straight elbow.
        cosine = (
            upper_x * forearm_x + upper_y * forearm_y + upper_z * forearm_z
        ) / (upper_length * forearm_length)
        elbow_bend = math.acos(max(-1.0, min(1.0, cosine)))
        elbow_bend = min(elbow_bend, 1.7)  # Lamp's usable bending range.
        self.arm_points = [
            (int(point.x * width), int(point.y * height)) for point in selected_points
        ]
        return self.smooth_angles(yaw, elevation, elbow_bend, now)

    def smooth_angles(self, yaw, elevation, elbow_bend, now):
        # One time-based filter reduces jitter without adding a second lag
        # in the display. Reset after losing sight of the arm for half a second.
        if self.last_seen is None or now - self.last_seen > self.reset_after_seconds:
            self.previous_angles = (yaw, elevation, elbow_bend)
        else:
            dt = max(0.0, now - self.last_seen)
            alpha = 1.0 - math.exp(-dt / self.smoothing_seconds)
            old_yaw, old_elevation, old_bend = self.previous_angles
            self.previous_angles = (
                old_yaw + alpha * (yaw - old_yaw),
                old_elevation + alpha * (elevation - old_elevation),
                old_bend + alpha * (elbow_bend - old_bend),
            )
        self.last_seen = now
        return self.previous_angles

    def annotate(self, frame):
        if self.arm_points is not None:
            shoulder, elbow, wrist = self.arm_points
            cv2.line(frame, shoulder, elbow, (0, 220, 255), 3)
            cv2.line(frame, elbow, wrist, (0, 220, 255), 3)
            cv2.circle(frame, shoulder, 9, (0, 255, 0), -1)
            cv2.circle(frame, elbow, 9, (255, 180, 0), -1)
            cv2.circle(frame, wrist, 9, (0, 220, 255), -1)
        status = "tracking" if self.arm_points is not None else "not visible - holding pose"
        cv2.putText(
            frame, f"{self.arm.title()} arm: {status} | A: switch arm",
            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1,
            cv2.LINE_AA,
        )
        return frame

    def close(self):
        self.landmarker.close()
