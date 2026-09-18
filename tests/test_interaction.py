import unittest
from types import SimpleNamespace

import numpy as np

from character.interaction import perform_actions, plan_request
from memory.scene_memory import SceneMemory
from robot.lamp import LampController


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.visible = {
            "bottle": {"position": "left", "confidence": 0.9},
            "person": {"position": "center", "confidence": 0.8},
        }
        self.remembered = {
            "bottle": {"position": "left", "last_seen": 10},
            "person": {"position": "center", "last_seen": 11},
        }

    def test_scene_question_uses_memory(self):
        actions, reply, target = plan_request(
            "What objects do you remember?", self.visible, self.remembered
        )
        self.assertEqual(actions, [])
        self.assertIn("bottle", reply)
        self.assertIsNone(target)

    def test_plural_object_name_matches_detector_label(self):
        actions, reply, target = plan_request(
            "Find the bottles and turn on the light", self.visible, self.remembered
        )
        self.assertEqual(target, "bottle")
        self.assertEqual(actions, [("look", "left"), ("light", "on")])
        self.assertEqual(reply, "")

    def test_goal_requires_target_to_be_visible_now(self):
        actions, reply, target = plan_request(
            "Find the bottle", {}, self.remembered
        )
        self.assertEqual(actions, [])
        self.assertIn("cannot see it now", reply)
        self.assertIsNone(target)

    def test_direct_light_command_is_an_action(self):
        actions, reply, target = plan_request(
            "Turn off your lamp", self.visible, self.remembered
        )
        self.assertEqual(actions, [("light", "off")])
        self.assertEqual(reply, "Turning the light off.")
        self.assertIsNone(target)

    def test_action_vocabulary_calls_lamp(self):
        calls = []
        lamp = SimpleNamespace(
            look_left=lambda: calls.append("left"),
            look_right=lambda: calls.append("right"),
            look_center=lambda: calls.append("center"),
            turn_on=lambda: calls.append("on"),
            turn_off=lambda: calls.append("off"),
        )
        perform_actions(lamp, [("look", "right"), ("light", "off")])
        self.assertEqual(calls, ["right", "off"])

    def test_look_action_moves_the_whole_lamp(self):
        lamp = LampController()

        lamp.follow_arm(1.5, 0.2, 1.4)
        lamp.look_left()

        self.assertEqual(lamp.base_yaw, 0.75)
        self.assertEqual(lamp.shoulder_pitch, 0.4)
        self.assertEqual(lamp.elbow_pitch, -0.8)
        self.assertEqual(lamp.neck_yaw, 0.25)


class SceneMemoryTests(unittest.TestCase):
    def test_detections_become_positions_and_memory(self):
        memory = SceneMemory()
        detections = SimpleNamespace(
            xyxy=np.array([[0, 0, 100, 20], [250, 0, 350, 20], [540, 0, 640, 20]]),
            class_id=np.array([0, 1, 2]),
            confidence=np.array([0.7, 0.8, 0.9]),
        )
        visible = memory.remember_detections(
            detections, {0: "cup", 1: "book", 2: "bottle"}, 640, 12.5
        )
        self.assertEqual(visible["cup"]["position"], "left")
        self.assertEqual(visible["book"]["position"], "center")
        self.assertEqual(visible["bottle"]["position"], "right")
        self.assertEqual(memory.get_objects()["bottle"]["last_seen"], 12.5)


if __name__ == "__main__":
    unittest.main()
