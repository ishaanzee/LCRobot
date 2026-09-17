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
