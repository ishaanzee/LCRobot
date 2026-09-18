import threading

import numpy as np
import sounddevice as sd


AUDIO_LOCK = threading.Lock()


class CharacterAudio:
    """Play short synthesized cues without blocking the camera loop."""

    def __init__(self, sample_rate=44_100):
        self.sample_rate = sample_rate

    def play_transition(self, previous_state, new_state):
        cues = {
            # Short sound effects acknowledge engagement and listening.
            "engaged": [(659, 0.10), (880, 0.18)],
            "listening": [(988, 0.14)],
            # An eight-note success theme is the character's music moment.
            "acting": [
                (523, 0.12), (659, 0.12), (784, 0.12), (659, 0.12),
                (698, 0.12), (784, 0.12), (880, 0.12), (1047, 0.28),
            ],
        }

        notes = cues.get(new_state)
        if new_state == "idle" and previous_state != "idle":
            # A descending phrase communicates a friendly disengagement.
            notes = [(659, 0.12), (523, 0.12), (392, 0.22)]

        if notes:
            threading.Thread(
                target=self._play_notes,
                args=(notes,),
                daemon=True,
            ).start()

    def _play_notes(self, notes):
        # Skip a new cue if another transition sound is still playing.
        if not AUDIO_LOCK.acquire(blocking=False):
            return

        try:
            audio = []
            for frequency, duration in notes:
                sample_count = int(self.sample_rate * duration)
                times = np.arange(sample_count) / self.sample_rate
                tone = 0.18 * np.sin(2 * np.pi * frequency * times)

                # A short fade prevents clicks at the edges of each note.
                fade_count = min(int(self.sample_rate * 0.01), sample_count // 2)
                if fade_count:
                    fade = np.linspace(0.0, 1.0, fade_count)
                    tone[:fade_count] *= fade
                    tone[-fade_count:] *= fade[::-1]

                audio.append(tone.astype(np.float32))
                audio.append(np.zeros(int(self.sample_rate * 0.025), dtype=np.float32))

            sd.play(np.concatenate(audio), self.sample_rate)
            sd.wait()
        except sd.PortAudioError as error:
            # Keep the character running if no speaker is available.
            print(f"Audio cue unavailable: {error}")
        finally:
            AUDIO_LOCK.release()
