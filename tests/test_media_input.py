import unittest
from types import SimpleNamespace

from media_input import ImageTooLargeError, decode_heif_image


class MediaInputTests(unittest.TestCase):
    def test_decodes_heif_image_and_converts_color(self):
        heif = SimpleNamespace(mode="RGB", size=(2, 1), data=b"heif-data")
        decoded_image = object()
        bgr_image = object()
        calls = {}

        def reader(path):
            calls["path"] = path
            return heif

        def image_factory(mode, size, data, decoder):
            calls["image_args"] = (mode, size, data, decoder)
            return decoded_image

        def converter(image):
            calls["converted"] = image
            return bgr_image

        result = decode_heif_image(
            "sample.heic",
            reader=reader,
            image_factory=image_factory,
            converter=converter,
        )

        self.assertIs(result, bgr_image)
        self.assertEqual(calls["path"], "sample.heic")
        self.assertEqual(calls["image_args"], ("RGB", (2, 1), b"heif-data", "raw"))
        self.assertIs(calls["converted"], decoded_image)

    def test_hides_reader_error_details_from_validation_message(self):
        def reader(_path):
            raise OSError("private decoder path /tmp/secret.heic")

        with self.assertRaisesRegex(ValueError, "Could not read image") as context:
            decode_heif_image(
                "sample.heic",
                reader=reader,
                image_factory=lambda *_args: object(),
                converter=lambda image: image,
            )

        self.assertNotIn("secret", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, OSError)

    def test_hides_image_conversion_error_details(self):
        heif = SimpleNamespace(mode="RGB", size=(2, 1), data=b"invalid")

        with self.assertRaisesRegex(ValueError, "valid supported image"):
            decode_heif_image(
                "sample.heic",
                reader=lambda _path: heif,
                image_factory=lambda *_args: (_ for _ in ()).throw(
                    ValueError("decoder internals")
                ),
                converter=lambda image: image,
            )


    def test_rejects_oversized_heif_before_creating_image_buffer(self):
        heif = SimpleNamespace(mode="RGB", size=(5001, 8000), data=b"large-image")
        factory_called = False

        def image_factory(*_args):
            nonlocal factory_called
            factory_called = True
            return object()

        with self.assertRaisesRegex(ImageTooLargeError, "40-megapixel"):
            decode_heif_image(
                "large.heic",
                reader=lambda _path: heif,
                image_factory=image_factory,
                converter=lambda image: image,
                max_pixels=40_000_000,
            )

        self.assertFalse(factory_called)

    def test_accepts_heif_at_the_configured_pixel_limit(self):
        heif = SimpleNamespace(mode="RGB", size=(4000, 10000), data=b"boundary-image")
        decoded_image = object()
        converted_image = object()

        result = decode_heif_image(
            "boundary.heic",
            reader=lambda _path: heif,
            image_factory=lambda *_args: decoded_image,
            converter=lambda image: converted_image if image is decoded_image else None,
            max_pixels=40_000_000,
        )

        self.assertIs(result, converted_image)


if __name__ == "__main__":
    unittest.main()
