# Local Validation Checklist

Use this checklist before changing the detection pipeline.

## Upload handling
- Test JPEG, PNG, WebP, HEIC/HEIF, and supported video inputs.
- Test an empty upload and a filename with unusual characters.
- Confirm unsupported extensions return a clear 400 response.
- Confirm generated files are removed after the cleanup window.

## Detection
- Verify confidence values at 0, 0.5, and 1.0.
- Verify malformed confidence input is rejected.
- Verify custom object lists are trimmed and lower-cased.
- Compare image and video output paths independently.

## AI runtime
- Start the application once and confirm both models load successfully.
- Check `/health` before a detection request.
- Run one CPU-only smoke test when GPU acceleration is unavailable.

## Regression evidence
Record the input type, confidence threshold, model availability, response status, and any user-visible error for each regression.
