import time
from pathlib import Path

import cv2


class EngagementDetector:
    def __init__(self, engage_delay=0.5, disengage_delay=1.0):
        cascade_path = Path(__file__).parent
        self.face_detector = cv2.CascadeClassifier(
            str(cascade_path / "haarcascade_frontalface_default.xml")
        )
        self.eye_detector = cv2.CascadeClassifier(
            str(cascade_path / "haarcascade_eye_tree_eyeglasses.xml")
        )

        self.engage_delay = engage_delay
        self.disengage_delay = disengage_delay
        self.engaged = False
        self.pending_state = None
        self.pending_since = None

    def detect(self, frame):
        looking = self._is_looking(frame)
        self._update_state(looking, time.monotonic())
        return self.engaged

    def _is_looking(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for x, y, width, height in faces:
            # Eyes should be in the upper portion of a detected face.
            face_top = gray[y : y + int(height * 0.65), x : x + width]
            eyes = self.eye_detector.detectMultiScale(
                face_top,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20),
            )
            if len(eyes) >= 1:
                return True

        return False

    def _update_state(self, looking, now):
        if looking == self.engaged:
            self.pending_state = None
            self.pending_since = None
            return

        if looking != self.pending_state:
            self.pending_state = looking
            self.pending_since = now
            return

        delay = self.engage_delay if looking else self.disengage_delay
        if now - self.pending_since >= delay:
            self.engaged = looking
            self.pending_state = None
            self.pending_since = None
