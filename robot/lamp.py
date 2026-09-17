class LampController:
    def __init__(self):
        self.light_on = False
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
        self.neck_yaw = 0.6
        self.head_pitch = 0.0
        print("Lamp pose: LOOK LEFT")

    def look_right(self):
        self.neck_yaw = -0.6
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

    def neutral(self):
        self.neck_yaw = 0.0
        self.head_pitch = 0.0
        print("Lamp pose: NEUTRAL")

    def follow_hand(self, x, y):
        # Keep commands within the normalized range expected from ArmTracker.
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))

        self.neck_yaw = -0.8 * x
        self.head_pitch = -0.5 * y
