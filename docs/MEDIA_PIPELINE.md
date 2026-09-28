# Vision Media Pipeline

The detection route validates request fields, sanitizes the original filename, saves a uniquely prefixed temporary input, and then runs image or video inference based on the file extension.

For images, OpenCV/Pillow-compatible processing produces an annotated result. Video requests are processed frame-by-frame before the output is written. The result, detection summary, and optional speech output are returned to the browser.

## Safety boundary

Treat uploaded filenames, confidence values, annotation colors, and VQA input filenames as untrusted request data. Keep path construction inside the configured upload/output folders and reject invalid values before inference or file access.

## Cleanup

The cleanup scheduler removes temporary artifacts after the configured delay. If a new output type is introduced, add it to the cleanup list before shipping.
