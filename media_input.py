from collections.abc import Callable
from typing import Any


class ImageTooLargeError(ValueError):
    """Raised when a decoded image exceeds the configured pixel budget."""


def decode_heif_image(
    path: str,
    *,
    reader: Callable[[str], Any],
    image_factory: Callable[..., Any],
    converter: Callable[[Any], Any],
    max_pixels: int | None = None,
) -> Any:
    """Decode a HEIF image without allocating an oversized PIL image buffer."""
    try:
        heif = reader(path)
        if max_pixels is not None:
            if isinstance(max_pixels, bool) or not isinstance(max_pixels, int) or max_pixels <= 0:
                raise ValueError("max_pixels must be a positive integer")

            size = getattr(heif, "size", None)
            if not isinstance(size, (tuple, list)) or len(size) != 2:
                raise ValueError("HEIF image dimensions are invalid")

            width, height = size
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (width, height)):
                raise ValueError("HEIF image dimensions are invalid")
            if width <= 0 or height <= 0:
                raise ValueError("HEIF image dimensions are invalid")
            if width * height > max_pixels:
                raise ImageTooLargeError(
                    "Image dimensions exceed the 40-megapixel processing limit."
                )

        image = image_factory(heif.mode, heif.size, heif.data, "raw")
        return converter(image)
    except ImageTooLargeError:
        raise
    except Exception as error:
        raise ValueError(
            "Could not read image. Please upload a valid supported image."
        ) from error
