import io
import os
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from groq import Groq

from character.audio import AUDIO_LOCK


def load_api_key(env_path):
    """Read only GROQ_API_KEY from the project's simple .env file."""
    if not env_path.exists():
        return None

    with env_path.open(encoding="utf-8") as env_file:
        for line in env_file:
            name, separator, value = line.partition("=")
            if separator and name.strip() == "GROQ_API_KEY":
                return value.strip().strip("'\"")
    return None


class Speech:
    def __init__(self, record_seconds=3):
        project_env = Path(__file__).resolve().parents[1] / ".env"
        api_key = os.getenv("GROQ_API_KEY") or load_api_key(project_env)
        if not api_key:
            raise RuntimeError("Add GROQ_API_KEY to the project .env file.")

        self.client = Groq(api_key=api_key)
        self.record_seconds = record_seconds
        self.sample_rate = 16_000

    def listen(self):
        print(f"Listening for {self.record_seconds} seconds...")
        with AUDIO_LOCK:
            recording = sd.rec(
                int(self.record_seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()

        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # int16 audio uses two bytes per sample.
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(recording.tobytes())

        transcription = self.client.audio.transcriptions.create(
            file=("microphone.wav", audio.getvalue()),
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en",
        )
        return transcription.text.strip()

    def speak(self, text):
        text = text.strip()
        if not text:
            return
        if len(text) > 200:
            raise ValueError("Speech text must be 200 characters or fewer.")

        response = self.client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice="autumn",
            input=text,
            response_format="wav",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "speech.wav"
            response.write_to_file(audio_path)
            with AUDIO_LOCK:
                self._play_wav(audio_path)

    def _play_wav(self, audio_path):
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            samples = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)

        if sample_width != 2:
            raise RuntimeError("Expected 16-bit WAV audio from the speech API.")
        if channels > 1:
            samples = samples.reshape(-1, channels)

        sd.play(samples, sample_rate)
        sd.wait()
