# Detection Failure Cleanup Checks

Use a temporary upload directory when running these checks. Cleanup is intentionally delayed by 15 minutes so a client can inspect the request result without retaining failed artifacts indefinitely.

## Manual regression cases

1. Upload a supported image with corrupt image bytes. Confirm the request returns a client error and the saved input is removed after the cleanup delay.
2. Upload a video with unreadable metadata. Confirm the request returns a client error and its saved input is eventually removed.
3. Upload a valid image, then simulate an exception during inference or result rendering. Confirm the request returns the generic internal-error response and any input, partial output, or partial audio artifact is scheduled for cleanup.
4. Complete a successful image and video request. Confirm the input, rendered output, and generated audio remain available during the normal 15-minute viewing window, then are removed.
5. Confirm the health endpoint remains responsive while cleanup threads are waiting.

These are manual integration checks; the repository's heavyweight model initialization means they should be run in an environment with the detector and VQA model dependencies installed.
