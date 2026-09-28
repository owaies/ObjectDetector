# Vision Model Runtime

The Flask application loads the YOLO-World detector and BLIP VQA model once when the process starts. This avoids repeatedly loading heavyweight model weights for each request.

## Request paths

- `/` renders the browser application.
- `/detect` handles image and video object detection.
- `/vqa` handles visual question answering.

Temporary uploads, generated media, and generated audio are stored under the configured static directories and are scheduled for cleanup in a background daemon thread.

## Maintenance guidance

Model loading belongs at application startup. Changes to model names or processor classes should be reviewed against `requirements.txt` and the deployment environment before release.
