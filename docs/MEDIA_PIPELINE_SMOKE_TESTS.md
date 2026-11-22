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

10. Submit a valid uploaded image to the VQA endpoint, then repeat with an image whose decoded dimensions exceed 40 megapixels. Confirm the oversized image receives HTTP 413 before image conversion or model inference.

11. Force a video inference error after at least one frame has been read. Confirm both the capture and writer are released before temporary-file cleanup is scheduled, and verify a later request can open another video.
12. Submit a video with invalid metadata and a video whose writer cannot be initialized. Confirm the capture is released on both early-return paths.

13. Simulate `cv2.imwrite` returning false and confirm the request reports a processing failure instead of rendering a result page with a missing image.
14. Submit a video that opens but yields no readable frames. Confirm the endpoint returns HTTP 400, releases the capture and writer, and does not render an empty output video.
