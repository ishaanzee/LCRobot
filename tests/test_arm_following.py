import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco

from robot.display import RobotDisplay
from robot.lamp import LampController
from vision.arm_tracker import ArmTracker


def tracker_without_camera(arm="right"):
    # Exercise the coordinate/filter methods without loading an ML model.
    tracker = ArmTracker.__new__(ArmTracker)
    tracker.arm = arm
    tracker.previous_angles = None
    tracker.last_seen = None
    tracker.arm_points = None
    tracker.smoothing_seconds = 0.08
    tracker.reset_after_seconds = 0.5
    return tracker


def pose(upper=(-0.3, 0, 0), forearm=(0, -0.3, 0), arm="right"):
    image = [SimpleNamespace(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]
    world = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(33)]
    shoulder, elbow, wrist = (12, 14, 16) if arm == "right" else (11, 13, 15)
    world[elbow] = SimpleNamespace(x=upper[0], y=upper[1], z=upper[2])
    world[wrist] = SimpleNamespace(
        x=upper[0] + forearm[0], y=upper[1] + forearm[1], z=upper[2] + forearm[2]
    )
    return image, world


def measure(tracker, image, world, now=0, width=640, height=480):
    return tracker.angles_from_landmarks(image, world, width, height, now)


class ArmTrackingTests(unittest.TestCase):
    def test_upper_arm_direction_controls_yaw_and_elevation(self):
        for upper, expected_yaw, expected_elevation in (
            ((0, 0, -0.3), 0, 0),             # Forward
            ((-0.3, 0, 0), math.pi / 2, 0),   # Image-left
            ((0.3, 0, 0), -math.pi / 2, 0),   # Image-right
            ((0, -0.3, -0.3), 0, math.pi / 4),
            ((0, 0.3, -0.3), 0, -math.pi / 4),
        ):
            with self.subTest(upper=upper):
                result = measure(tracker_without_camera(), *pose(upper=upper))
                self.assertAlmostEqual(result[0], expected_yaw)
                self.assertAlmostEqual(result[1], expected_elevation)

    def test_forearm_movement_leaves_shoulder_unchanged(self):
        straight = measure(tracker_without_camera(), *pose(forearm=(-0.3, 0, 0)))
        bent = measure(tracker_without_camera(), *pose(forearm=(0, -0.3, 0)))
        self.assertEqual(straight[:2], bent[:2])
        self.assertAlmostEqual(straight[2], 0)
        self.assertAlmostEqual(bent[2], math.pi / 2)

    def test_elbow_angle_uses_depth_on_either_arm(self):
        for arm in ("left", "right"):
            for forearm, expected_bend in (
                ((0, 0, -0.3), 0), ((0, -0.3, 0), math.pi / 2),
            ):
                result = measure(tracker_without_camera(arm), *pose(
                    upper=(0, 0, -0.3), forearm=forearm, arm=arm
                ))
                self.assertAlmostEqual(result[2], expected_bend)

    def test_arm_switch_clears_previous_angles(self):
        tracker = tracker_without_camera()
        measure(tracker, *pose())
        tracker.switch_arm()
        self.assertEqual(tracker.arm, "left")
        self.assertIsNone(tracker.previous_angles)
        self.assertIsNone(tracker.arm_points)
        result = measure(tracker, *pose(upper=(0.3, 0, 0), arm="left"), now=0.1)
        self.assertAlmostEqual(result[0], -math.pi / 2)

    def test_translation_scale_and_image_size_do_not_change_angles(self):
        image, world = pose()
        first = measure(tracker_without_camera(), image, world)
        moved = [SimpleNamespace(
            x=p.x * 1.5 + 1, y=p.y * 1.5 + 2, z=p.z * 1.5 - 3
        ) for p in world]
        second = measure(tracker_without_camera(), image, moved, width=1280, height=720)
        for a, b in zip(first, second):
            self.assertAlmostEqual(a, b)

    def test_invalid_landmarks_hold_last_pose(self):
        for problem in ("hidden_shoulder", "hidden_elbow", "hidden_wrist",
                        "offscreen", "overlap", "nan", "infinity"):
            with self.subTest(problem=problem):
                tracker = tracker_without_camera()
                image, world = pose()
                previous = measure(tracker, image, world)
                if problem.startswith("hidden"):
                    index = {"hidden_shoulder": 12, "hidden_elbow": 14, "hidden_wrist": 16}[problem]
                    image[index].visibility = 0.1
                elif problem == "offscreen":
                    image[14].x = 1.2
                elif problem == "overlap":
                    world[14] = world[12]
                elif problem == "nan":
                    world[14].z = float('nan')
                else:
                    world[14].x = float('inf')
                self.assertIsNone(measure(tracker, image, world, now=0.1))
                self.assertEqual(tracker.previous_angles, previous)
                self.assertIsNone(tracker.arm_points)

    def test_nearly_vertical_arm_holds_last_yaw(self):
        tracker = tracker_without_camera()
        previous = measure(tracker, *pose())
        for upper in ((0.001, -0.3, 0), (-0.001, -0.3, 0.001)):
            result = measure(tracker, *pose(upper=upper), now=0.1)
            self.assertEqual(result[0], previous[0])
        # With no recent direction, a vertical arm starts facing forward.
        result = measure(tracker_without_camera(), *pose(upper=(0, -0.3, 0)))
        self.assertEqual(result[0], 0)
        self.assertAlmostEqual(result[1], math.pi / 2)

    def test_yaw_does_not_flip_at_rear_angle_boundary(self):
        for side in (-1, 1):
            tracker = tracker_without_camera()
            previous = measure(tracker, *pose(upper=(side * 0.001, 0, 0.3)))
            result = measure(tracker, *pose(upper=(-side * 0.001, 0, 0.3)), now=0.1)
            self.assertAlmostEqual(result[0], previous[0])
            self.assertAlmostEqual(abs(result[0]), 2.45)

    def test_filter_response_is_independent_of_frame_rate(self):
        responses = []
        for fps in (15, 30, 60):
            tracker = tracker_without_camera()
            tracker.smooth_angles(0, 0, 0, 0)
            for frame in range(1, fps // 5 + 1):
                result = tracker.smooth_angles(1, -1, 1, frame / fps)
            self.assertAlmostEqual(result[2], result[0])
            responses.append(result[0])
            self.assertGreater(result[0], 0.9)
            self.assertLess(result[0], 1)
        self.assertAlmostEqual(responses[0], responses[1])
        self.assertAlmostEqual(responses[1], responses[2])

    def test_reacquisition_does_not_blend_with_old_pose(self):
        tracker = tracker_without_camera()
        measure(tracker, *pose())
        new_pose = pose(upper=(0.3, 0, 0))
        result = measure(tracker, *new_pose, now=1)
        expected = measure(tracker_without_camera(), *new_pose, now=1)
        self.assertEqual(result, expected)


class LampMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "robot/dummy_lamp_5dof.urdf"
        cls.model = mujoco.MjModel.from_xml_path(str(path))

    def apply_pose(self, lamp, data):
        for name, attribute in RobotDisplay.JOINTS.items():
            joint = self.model.joint(name)
            angle = getattr(lamp, attribute)
            self.assertGreaterEqual(angle, joint.range[0])
            self.assertLessEqual(angle, joint.range[1])
            data.qpos[joint.qposadr[0]] = angle
        mujoco.mj_forward(self.model, data)

    def test_upper_arm_elevation_raises_lamp_lower_segment(self):
        lamp = LampController()
        data = mujoco.MjData(self.model)
        for yaw in (-2, 0, 2):
            last_height = float('-inf')
            for step in range(21):
                elevation = -math.pi / 2 + step * math.pi / 20
                lamp.follow_arm(yaw, elevation, 0.8)
                self.apply_pose(lamp, data)
                height = data.body('upper_arm_link').xpos[2]  # Middle pivot
                self.assertGreaterEqual(height + 1e-9, last_height)
                last_height = height

    def test_lamp_segment_matches_reachable_upper_arm_angle(self):
        lamp = LampController()
        data = mujoco.MjData(self.model)
        minimum_elevation = math.pi / 2 - 0.95
        for elevation in (minimum_elevation, math.pi / 4, math.pi / 2):
            lamp.follow_arm(0.7, elevation, 0)
            self.apply_pose(lamp, data)
            shoulder = data.body("lower_arm_link").xpos
            middle_pivot = data.body("upper_arm_link").xpos
            segment = middle_pivot - shoulder
            measured_elevation = math.atan2(
                segment[2], math.hypot(segment[0], segment[1])
            )
            self.assertAlmostEqual(measured_elevation, elevation)

    def test_arm_below_lamp_range_stops_at_soft_limit(self):
        lamp = LampController()
        lamp.follow_arm(0, 0, 0)
        self.assertEqual(lamp.shoulder_pitch, 0.95)

    def test_base_rotates_lower_arm_in_expected_direction(self):
        lamp = LampController()
        data = mujoco.MjData(self.model)
        for yaw in (-math.pi / 2, 0, math.pi / 2):
            lamp.follow_arm(yaw, 0, 0)
            self.apply_pose(lamp, data)
            # Middle-pivot x/y position reveals the lower arm's heading.
            x, y, _ = data.body('upper_arm_link').xpos
            self.assertAlmostEqual(math.atan2(y, x), yaw)

    def test_middle_pivot_follows_elbow_independently_of_shoulder(self):
        lamp = LampController()
        for yaw, elevation in ((-1, -1), (0, 0), (1, 1)):
            for bend in (0, math.pi / 2, 1.7):
                lamp.follow_arm(yaw, elevation, bend)
                self.assertAlmostEqual(lamp.elbow_pitch, -bend)
                self.assertEqual(lamp.neck_yaw, 0)
                self.assertEqual(lamp.head_pitch, 0)

    def test_extreme_input_stays_inside_urdf_limits(self):
        lamp = LampController()
        data = mujoco.MjData(self.model)
        for yaw in (-100, 0, 100):
            for elevation in (-100, 0, 100):
                for bend in (-100, 0, 100):
                    lamp.follow_arm(yaw, elevation, bend)
                    self.apply_pose(lamp, data)


if __name__ == "__main__":
    unittest.main()
