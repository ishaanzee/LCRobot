from concurrent.futures import ThreadPoolExecutor
import resource
import sys
import time


def peak_memory_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def main():
    # Spawned gesture workers import this module. Load the rest of the models
    # only in the main app so the worker does not also load PyTorch and MuJoCo.
    import cv2
    from rfdetr.assets.coco_classes import COCO_CLASSES

    from character.interaction import perform_actions, plan_request
    from character.state import CharacterState, CharacterStateMachine
    from memory.scene_memory import SceneMemory
    from robot.display import RobotDisplay
    from robot.lamp import LampController
    from speech.speech import Speech
    from vision.arm_tracker import ArmTracker
    from vision.detector import Detector
    from vision.engagement import EngagementDetector
    from vision.gesture_process import GestureProcess

    detector = Detector()
    engagement_detector = EngagementDetector()
    arm_tracker = ArmTracker()
    character = CharacterStateMachine()
    memory = SceneMemory()
    lamp = LampController()
    robot_display = RobotDisplay()
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not webcam.isOpened():
        arm_tracker.close()
        robot_display.close()
        raise RuntimeError("Could not open webcam at index 0")

    gesture_detector = GestureProcess()
    # Object recognition is slower than arm tracking. Allow only one job at
    # a time so old camera frames never build up in a background queue.
    object_worker = ThreadPoolExecutor(max_workers=1)
    voice_worker = ThreadPoolExecutor(max_workers=1)
    object_job = None
    object_frame_time = None
    detections = None
    detections_frame_time = None
    visible_objects = {}
    next_object_detection = 0.0
    speech = None
    voice_job = None
    voice_stage = None
    voice_started_at = None
    pending_goal = None
    last_words = "Press V to speak"
    report_started = time.monotonic()
    report_cpu_started = time.process_time()
    report_frames = 0
    frames_per_second = 0.0
    was_engaged = False
    try:
        while True:
            success, frame = webcam.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            now = time.monotonic()
            report_frames += 1
            report_elapsed = now - report_started
            if report_elapsed >= 5.0:
                frames_per_second = report_frames / report_elapsed
                cpu_percent = 100 * (
                    time.process_time() - report_cpu_started
                ) / report_elapsed
                print(
                    f"Performance: {frames_per_second:.1f} FPS, "
                    f"{cpu_percent:.0f}% CPU, {peak_memory_mb():.0f} MB peak RSS"
                )
                report_started = now
                report_cpu_started = time.process_time()
                report_frames = 0
            if object_job is not None and object_job.done():
                detections = object_job.result()
                detections_frame_time = object_frame_time
                visible_objects = memory.remember_detections(
                    detections, COCO_CLASSES, frame.shape[1], now
                )
                object_job = None
            if object_job is None and now >= next_object_detection:
                object_job = object_worker.submit(detector.detect, frame.copy())
                object_frame_time = now
                next_object_detection = now + 0.5
            annotated_frame = (
                detector.annotate(frame, detections)
                if detections is not None else frame.copy()
            )

            engaged = engagement_detector.detect(frame)
            interaction_busy = voice_stage is not None or pending_goal is not None
            if not interaction_busy:
                new_state = CharacterState.ENGAGED if engaged else CharacterState.IDLE
                character.transition_to(new_state)
                if engaged != was_engaged:
                    lamp.turn_on() if engaged else lamp.turn_off()
                    was_engaged = engaged

            if voice_job is not None and voice_job.done():
                try:
                    result = voice_job.result()
                    print(
                        f"{voice_stage.title()} latency: "
                        f"{now - voice_started_at:.2f} seconds"
                    )
                    if voice_stage == "listening":
                        last_words = result or "I did not hear anything."
                        print(f"You said: {last_words}")
                        character.transition_to(CharacterState.THINKING)
                        actions, reply, target = plan_request(
                            last_words, visible_objects, memory.get_objects()
                        )
                        if actions and target:
                            character.transition_to(CharacterState.ACTING)
                            perform_actions(lamp, actions)
                            pending_goal = {
                                "target": target,
                                "action_time": now,
                                # Hold the object-facing pose before checking
                                # the camera again. Arm following stays paused
                                # while a goal is pending.
                                "verify_after": now + 2.0,
                            }
                            voice_job = None
                            voice_stage = None
                        else:
                            if actions:
                                perform_actions(lamp, actions)
                            character.transition_to(CharacterState.RESPONDING)
                            voice_job = voice_worker.submit(speech.speak, reply)
                            voice_stage = "speaking"
                            voice_started_at = now
                    else:
                        voice_job = None
                        voice_stage = None
                except Exception as error:
                    print(f"Speech interaction failed: {error}")
                    last_words = "Speech failed - check the terminal"
                    voice_job = None
                    voice_stage = None

            fresh_detection = (
                pending_goal is not None
                and detections_frame_time is not None
                and detections_frame_time >= pending_goal["action_time"]
                and now >= pending_goal["verify_after"]
            )
            if fresh_detection:
                character.transition_to(CharacterState.OBSERVING)
                target = pending_goal["target"]
                action_time = pending_goal["action_time"]
                if target in visible_objects:
                    reply = f"I found the {target} and checked that it is still here."
                else:
                    reply = f"I moved toward the {target}, but I lost sight of it."
                pending_goal = None
                print(
                    f"Goal observation latency: "
                    f"{now - action_time:.2f} seconds"
                )
                character.transition_to(CharacterState.RESPONDING)
                voice_job = voice_worker.submit(speech.speak, reply)
                voice_stage = "speaking"
                voice_started_at = now

            arm_pose = arm_tracker.detect(frame)
            gesture = gesture_detector.detect(frame)
            if pending_goal is None and voice_stage != "speaking":
                if gesture == "on":
                    lamp.turn_on()
                elif gesture == "off":
                    lamp.turn_off()
            if arm_pose is not None and pending_goal is None and voice_stage != "speaking":
                lamp.follow_arm(*arm_pose)
            arm_tracker.annotate(annotated_frame)
            cv2.putText(
                annotated_frame, f"Hand: {gesture_detector.status}",
                (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                f"State: {character.current.value}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if engaged else (180, 180, 180),
                2,
            )
            cv2.putText(
                annotated_frame,
                last_words[:70],
                (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"FPS: {frames_per_second:.1f}",
                (535, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            vision_frame = cv2.resize(annotated_frame, (640, 480))
            robot_frame = cv2.cvtColor(
                robot_display.render(lamp),
                cv2.COLOR_RGB2BGR,
            )
            cv2.putText(
                robot_frame,
                "V: speak | A: switch arm | Q: quit",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            combined_frame = cv2.hconcat([vision_frame, robot_frame])
            cv2.imshow("Character Robot", combined_frame)

            # waitKey also lets OpenCV refresh the display window.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("a"):
                arm_tracker.switch_arm()
            elif key == ord("v") and voice_job is None and pending_goal is None:
                try:
                    speech = speech or Speech()
                    character.transition_to(CharacterState.LISTENING)
                    last_words = "Listening for 3 seconds..."
                    voice_job = voice_worker.submit(speech.listen)
                    voice_stage = "listening"
                    voice_started_at = time.monotonic()
                except RuntimeError as error:
                    last_words = str(error)
                    print(error)
            elif key == ord("q"):
                break
    finally:
        gesture_detector.close()
        object_worker.shutdown(wait=True, cancel_futures=True)
        voice_worker.shutdown(wait=True, cancel_futures=True)
        arm_tracker.close()
        robot_display.close()
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
