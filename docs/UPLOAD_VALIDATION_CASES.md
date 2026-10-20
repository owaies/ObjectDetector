# Media Upload Boundary Cases

When modifying the upload handlers, retain coverage for these input classes:

- No file field and an empty filename.
- A supported extension with invalid file bytes.
- A valid image whose decoded pixel count exceeds the application limit.
- An object filter containing only whitespace or more unique labels than the supported maximum.
- A filename containing path separators or unusual Unicode characters.
- A video with no decodable frames.
- A valid small fixture that produces a normal result.

Check both the HTTP status and the response body. Avoid committing uploaded samples or generated outputs to the repository.
