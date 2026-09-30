# API Smoke Tests

The application exposes a small HTTP surface that can be checked without running a full end-to-end browser test.

## Health

```bash
curl -i http://localhost:5000/health
```

Expected response:

```json
{"status":"ok","service":"object-detector"}
```

## Detection validation

Send a multipart request with an image and optional form fields:

- `image`: uploaded image/video file
- `objects`: comma-separated target classes
- `confidence`: number from 0 to 1
- `color`: six-digit hexadecimal bounding-box color

Check these cases separately:

1. Missing `image` returns HTTP 400.
2. Empty filenames return HTTP 400.
3. Invalid confidence values return HTTP 400.
4. Confidence outside 0..1 returns HTTP 400.
5. Invalid colors return HTTP 400.
6. A valid request returns the generated result according to the input media type.

Keep these checks aligned with the validation order in `app.py` so future changes do not silently weaken the request boundary.
