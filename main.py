import cv2

from vision.detector import Detector


def main():
    detector = Detector()
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        raise RuntimeError("Could not open webcam at index 0")

    try:
        while True:
            success, frame = webcam.read()
            if not success:
                print("Could not read a frame from the webcam.")
                break

            detections = detector.detect(frame)
            annotated_frame = detector.annotate(frame, detections)
            cv2.imshow("Character Robot Vision", annotated_frame)

            # waitKey also lets OpenCV refresh the display window.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
