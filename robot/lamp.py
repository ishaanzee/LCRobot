import math


class LampController:
    def __init__(self):
        self.light_on = False
        self.base_yaw = 0.0
        self.shoulder_pitch = 0.25
        self.elbow_pitch = -0.85
        self.neck_yaw = 0.0
        self.head_pitch = 0.0

    def turn_on(self):
        if self.light_on:
            return

        self.light_on = True
        print("Lamp light: ON")

    def turn_off(self):
        if not self.light_on:
            return

        self.light_on = False
        print("Lamp light: OFF")

    def look_left(self):
        self.base_yaw = 0.75
        self.shoulder_pitch = 0.4
        self.elbow_pitch = -0.8
        self.neck_yaw = 0.25
        self.head_pitch = 0.0
        print("Lamp pose: LOOK LEFT")

    def look_right(self):
        self.base_yaw = -0.75
        self.shoulder_pitch = 0.4
        self.elbow_pitch = -0.8
        self.neck_yaw = -0.25
        self.head_pitch = 0.0
        print("Lamp pose: LOOK RIGHT")

    def look_up(self):
        self.neck_yaw = 0.0
        self.head_pitch = -0.4
        print("Lamp pose: LOOK UP")

    def look_down(self):
        self.neck_yaw = 0.0
        self.head_pitch = 0.4
        print("Lamp pose: LOOK DOWN")

    def look_center(self):
        self.base_yaw = 0.0
        self.shoulder_pitch = 0.4
        self.elbow_pitch = -0.8
        self.neck_yaw = 0.0
        self.head_pitch = 0.0
        print("Lamp pose: LOOK CENTER")

    def neutral(self):
        self.base_yaw = 0.0
        self.shoulder_pitch = 0.25
        self.elbow_pitch = -0.85
        self.neck_yaw = 0.0
        self.head_pitch = 0.0
        print("Lamp pose: NEUTRAL")

    def follow_arm(self, shoulder_yaw, arm_elevation, elbow_bend):
        # Base yaw and shoulder pitch together imitate two shoulder rotations.
        self.base_yaw = max(-2.45, min(2.45, shoulder_yaw))
        arm_elevation = max(-math.pi / 2, min(math.pi / 2, arm_elevation))
        # The lamp starts vertical, so its joint angle is 90 degrees minus the
        # upper arm's angle above the ground. Clamp at the URDF soft limit: the
        # lamp cannot safely become horizontal or point through the table.
        desired_pitch = math.pi / 2 - arm_elevation
        self.shoulder_pitch = max(0.0, min(0.95, desired_pitch))
        self.elbow_pitch = -max(0.0, min(1.7, elbow_bend))
        # Keep the head aligned with the upper lamp segment during imitation.
        self.neck_yaw = 0.0
        self.head_pitch = 0.0
