import tempfile
import unittest
from pathlib import Path

from speech.speech import load_api_key


class SpeechConfigTests(unittest.TestCase):
    def test_loads_only_named_key_from_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "OTHER=value\nGROQ_API_KEY='test-key'\n",
                encoding="utf-8",
            )
            self.assertEqual(load_api_key(env_path), "test-key")

    def test_missing_file_has_no_key(self):
        self.assertIsNone(load_api_key(Path("does-not-exist")))


if __name__ == "__main__":
    unittest.main()
