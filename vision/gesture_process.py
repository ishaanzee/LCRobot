import multiprocessing
import time


def recognize_hands(connection):
    """Own MediaPipe entirely inside this process, including cleanup."""
    from vision.gestures import GestureDetector

    detector = None
    try:
        detector = GestureDetector()
        connection.send(("ready", None))
        while True:
            frame = connection.recv()
            if frame is None:
                break
            connection.send(("result", detector.detect(frame)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        connection.send(("error", str(error)))
    finally:
        if detector is not None:
            detector.close()
        connection.close()


class GestureProcess:
    """One frame in, one gesture out; native failures stay outside the app."""

    def __init__(self):
        self.context = multiprocessing.get_context("spawn")
        self.process = None
        self.connection = None
        self.status = "starting"
        self.waiting = True
        self.started_at = 0.0
        self.sent_at = 0.0
        self.retry_at = 0.0
        self.restarts = 0
        self.start()

    def start(self):
        self.connection, worker_connection = self.context.Pipe()
        self.process = self.context.Process(
            target=recognize_hands, args=(worker_connection,), daemon=True
        )
        self.process.start()
        worker_connection.close()
        self.started_at = time.monotonic()
        self.waiting = True
        self.status = "starting"

    def detect(self, frame):
        now = time.monotonic()
        if self.process is None:
            if self.restarts < 3 and now >= self.retry_at:
                self.restarts += 1
                self.start()
            return None

        gesture = None
        try:
            if not self.process.is_alive():
                raise RuntimeError(f"worker exited with code {self.process.exitcode}")
            if self.connection.poll():
                message, value = self.connection.recv()
                if message == "error":
                    raise RuntimeError(value)
                self.waiting = False
                if message == "result" and now - self.sent_at <= 0.75:
                    gesture = value
                self.status = {"on": "open", "off": "closed", None: "no gesture"}[gesture]

            # The child asks for another frame only after finishing the last.
            # This avoids a queue of old images while keeping the UI responsive.
            if not self.waiting:
                self.connection.send(frame)
                self.sent_at = now
                self.waiting = True
            elif now - max(self.started_at, self.sent_at) > 20:
                raise RuntimeError("worker stopped responding")
        except (EOFError, BrokenPipeError, OSError, RuntimeError) as error:
            print(f"Gesture recognition restarting: {error}")
            self.close()
            self.retry_at = now + 1
            self.status = "restarting" if self.restarts < 3 else "unavailable - restart app"
            return None
        return gesture

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                try:
                    self.connection.send(None)
                except (BrokenPipeError, OSError):
                    pass
                self.process.join(timeout=0.5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=0.5)
            else:
                self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
