class SceneMemory:
    def __init__(self):
        self.objects = {}

    def observe(self, label, position, timestamp):
        self.objects[label] = {
            "position": position,
            "last_seen": timestamp,
        }

    def get_objects(self):
        # Return copies so callers cannot accidentally change stored memories.
        return {
            label: observation.copy()
            for label, observation in self.objects.items()
        }

    def remember_detections(self, detections, class_names, frame_width, timestamp):
        """Store the strongest current observation for each detected object type."""
        visible = {}
        if detections is None:
            return visible

        for box, class_id, confidence in zip(
            detections.xyxy, detections.class_id, detections.confidence
        ):
            label = class_names[int(class_id)]
            center_x = float(box[0] + box[2]) / 2
            position = "left" if center_x < frame_width / 3 else (
                "right" if center_x > frame_width * 2 / 3 else "center"
            )
            previous = visible.get(label)
            if previous is None or confidence > previous["confidence"]:
                visible[label] = {
                    "position": position,
                    "confidence": float(confidence),
                }

        for label, observation in visible.items():
            self.observe(label, observation["position"], timestamp)
        return visible
