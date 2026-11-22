import tempfile
import unittest
from pathlib import Path

from media_output import ensure_nonempty_output, validate_video_output, write_image_output


class MediaOutputTests(unittest.TestCase):
    def test_image_output_requires_successful_writer_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = str(Path(directory) / "annotated.jpg")

            with self.assertRaisesRegex(OSError, "could not write"):
                write_image_output(output_path, object(), lambda *_: False)

            self.assertFalse(Path(output_path).exists())

    def test_image_output_requires_nonempty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = str(Path(directory) / "annotated.jpg")

            with self.assertRaisesRegex(OSError, "empty"):
                write_image_output(output_path, object(), lambda *_: Path(output_path).touch() or True)

    def test_image_output_accepts_nonempty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = str(Path(directory) / "annotated.jpg")

            def writer(path, _image):
                Path(path).write_bytes(b"image-data")
                return True

            write_image_output(output_path, object(), writer)
            self.assertGreater(Path(output_path).stat().st_size, 0)

    def test_video_output_rejects_zero_decoded_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = str(Path(directory) / "annotated.mp4")

            with self.assertRaisesRegex(ValueError, "No readable video frames"):
                validate_video_output(output_path, 0)

    def test_video_output_requires_nonempty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = str(Path(directory) / "annotated.mp4")
            Path(output_path).touch()

            with self.assertRaisesRegex(OSError, "empty"):
                validate_video_output(output_path, 1)

    def test_video_output_accepts_nonempty_file_with_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = str(Path(directory) / "annotated.mp4")
            Path(output_path).write_bytes(b"video-data")

            validate_video_output(output_path, 1)

    def test_missing_output_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = str(Path(directory) / "missing.jpg")

            with self.assertRaisesRegex(OSError, "not created"):
                ensure_nonempty_output(output_path)


if __name__ == "__main__":
    unittest.main()
