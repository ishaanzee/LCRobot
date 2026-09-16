import cv2
import supervision as sv
from rfdetr import RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES


class Detector:
    def __init__(self):
        self.model = RFDETRNano()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def detect(self, frame):
        # OpenCV uses BGR, while RF-DETR expects RGB images.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.model.predict(frame_rgb, threshold=0.5)

    def annotate(self, frame, detections):
        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(
                detections.class_id, detections.confidence
            )
        ]

        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        return self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )
