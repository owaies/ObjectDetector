from pathlib import Path
from typing import Callable


def ensure_nonempty_output(path: str) -> None:
    """Raise when a processed media output is missing or empty."""
    try:
        output_size = Path(path).stat().st_size
    except OSError as error:
        raise OSError("Processed output file was not created or cannot be read.") from error

    if output_size <= 0:
        raise OSError("Processed output file is empty.")


def write_image_output(path: str, image: object, writer: Callable[[str, object], bool]) -> None:
    """Write an annotated image and verify OpenCV completed the write."""
    if not writer(path, image):
        raise OSError("OpenCV could not write the processed image.")

    ensure_nonempty_output(path)


def validate_video_output(path: str, frames_written: int) -> None:
    """Reject videos with no decoded frames or no persisted output."""
    if frames_written <= 0:
        raise ValueError("No readable video frames were found.")

    ensure_nonempty_output(path)
