# Media Pipeline Smoke Tests

Use small, non-sensitive fixtures to validate the image and video processing flow.

1. Open the landing page and confirm model startup has completed before attempting analysis.
2. Submit a supported image and verify the response includes the expected detections and annotated output.
3. Submit a supported short video and confirm frame processing completes within the configured limits.
4. Try an unsupported extension, empty upload, and missing file field; each should produce a controlled client error.
5. Try a valid image with a very large decoded dimension and confirm the media-size guard rejects it before expensive processing.
6. Verify generated files are served from the expected output directory and temporary artifacts are eventually cleaned up.
7. Check the browser console and server log for unexpected exceptions.

Record fixture dimensions, processing time, and whether the test used CPU or GPU inference.

8. Submit two detection requests concurrently with different custom object lists. Confirm each response's labels correspond only to its own requested classes; a concurrent request must not change the class names used for another request's inference or label lookup.
9. Repeat the concurrent-request check with one request using the default object classes and the other using custom classes, then verify that a subsequent default request uses the complete default class list.
