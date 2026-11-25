from collections.abc import Callable
from typing import Any


def decode_heif_image(
    path: str,
    *,
    reader: Callable[[str], Any],
    image_factory: Callable[..., Any],
    converter: Callable[[Any], Any],
) -> Any:
    """Decode a HEIF image while keeping low-level decoder errors out of API responses."""
    try:
        heif = reader(path)
        image = image_factory(heif.mode, heif.size, heif.data, "raw")
        return converter(image)
    except Exception as error:
        raise ValueError(
            "Could not read image. Please upload a valid supported image."
        ) from error
