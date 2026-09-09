<div align="center">

# 🔬 AI Vision Suite

### Object Detection & Visual Question Answering

<p>
  <strong>Upload an image or video, detect objects with YOLO-World, and ask visual questions with BLIP.</strong>
</p>

<p>
  <a href="https://world-object-detector.netlify.app/"><strong>🌐 Live Website</strong></a> ·
  <a href="https://github.com/owaies/ObjectDetector">Repository</a> ·
  <a href="https://github.com/owaies/ObjectDetector/issues">Issues</a> ·
  <a href="https://github.com/owaies">Author</a>
</p>

</div>

---

## 🌐 Try it online

### [🚀 Open AI Vision Suite](https://world-object-detector.netlify.app/)

Use the deployed application directly in your browser:

**https://world-object-detector.netlify.app/**

---

## ✨ What is this?

**AI Vision Suite** is a Flask-based computer-vision web application that combines **YOLO-World object detection** with **BLIP Visual Question Answering (VQA)** in a simple browser interface.

The project supports image and video analysis, configurable confidence thresholds, custom detection classes, annotated output, and optional text-to-speech results. The application loads its AI models once when the server starts, then processes uploaded media through the Flask routes. citeturn4file0

> 🧠 **Think of it as a small vision lab:** give it media, choose what you want to inspect, and let the models turn pixels into useful answers.

---

## 🎯 Features

| Feature | Description |
|---|---|
| 🎯 **Object Detection** | Detect objects with `yolov8s-world.pt`. |
| 🧩 **Custom Classes** | Enter comma-separated objects such as `person, car`. |
| 🎚️ **Confidence Control** | Tune the detection threshold from 10% to 90%. |
| 🎥 **Video Processing** | Process `.mp4`, `.mov`, `.avi`, `.mkv`, and `.webm` videos. |
| 🖼️ **Image Processing** | Annotate supported images with bounding boxes and confidence scores. |
| 🍎 **HEIF Support** | Includes `pi-heif` handling for HEIF/HEIC image input. |
| ❓ **Visual Q&A** | Ask questions about an uploaded image using Salesforce BLIP VQA. |
| 🔊 **Text-to-Speech** | Convert detection summaries and VQA answers to speech with gTTS. |
| 🧹 **Automatic Cleanup** | Temporary uploaded, output, and audio files are scheduled for cleanup after 15 minutes. |
| 📱 **Interactive UI** | Bootstrap-based interface with mode controls, confidence slider, color picker, and processing feedback. |

The implemented backend exposes `/`, `/detect`, and `/vqa` routes, while the frontend provides detection-mode controls and an interactive confidence slider. citeturn4file0 citeturn6file0

---

## 🖥️ How it works

```text
                    ┌──────────────────────┐
                    │      Web Browser     │
                    │   Upload + Controls  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │      Flask App       │
                    │  /detect   /vqa      │
                    └───────┬───────┬──────┘
                            │       │
                 ┌──────────▼──┐ ┌─▼─────────────────┐
                 │ YOLO-World  │ │ Salesforce BLIP   │
                 │ Detection   │ │ Visual Q&A       │
                 └──────┬──────┘ └────────┬──────────┘
                        │                  │
                 ┌──────▼──────────────────▼──────┐
                 │ Annotated Media + AI Answers   │
                 │          + Optional Audio      │
                 └────────────────────────────────┘
```

### Detection flow

1. Upload an image or video.
2. Optionally enter target classes, for example `person, car, bicycle`.
3. Adjust the confidence threshold.
4. Choose the annotation color for detection output.
5. Submit the media for processing.
6. The application returns the original media, processed output, detection summary, and optional speech audio.

### VQA flow

For an uploaded image, the application can receive a natural-language question, run BLIP VQA, and return an answer plus optional speech audio. citeturn4file0

---

## 🗂️ Project structure

```text
ObjectDetector/
├── app.py                  # Flask application + AI inference routes
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── templates/
    ├── index.html          # Upload and analysis interface
    └── results.html        # Results presentation
```

The current repository contains these application files on the `main` branch. citeturn3file0

---

## ⚙️ Tech stack

- **Python**
- **Flask** for the web server
- **Ultralytics YOLO** for object detection
- **YOLO-World** weights via `yolov8s-world.pt`
- **OpenCV** for image/video processing
- **Transformers + PyTorch** for BLIP VQA
- **Pillow + pi-heif** for image handling
- **gTTS** for text-to-speech
- **Bootstrap 5** for the browser UI

The dependency list is defined in `requirements.txt`. citeturn5file0

---

## 🚀 Run locally

### 1. Clone the repository

```bash
git clone https://github.com/owaies/ObjectDetector.git
cd ObjectDetector
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Provide the detection weights

The application initializes `YOLO('yolov8s-world.pt')`, so the `yolov8s-world.pt` model file must be available to the application at runtime. citeturn4file0

### 5. Start the server

```bash
python app.py
```

The application listens on `0.0.0.0` and uses the `PORT` environment variable when provided, otherwise it defaults to **7860**. citeturn4file0

Then open:

```text
http://localhost:7860
```

> 💡 On first startup, the BLIP VQA processor/model are loaded from the Hugging Face model `Salesforce/blip-vqa-base`. citeturn4file0

---

## 🎛️ Using the web interface

### Object detection

1. Select **Bounding Box Detection**.
2. Upload an image or video.
3. Optionally enter the objects to search for.
4. Set the confidence threshold.
5. Pick a box color.
6. Click **Analyze**.

Example target list:

```text
person, car, bicycle, dog
```

The backend accepts the object list as a comma-separated value and uses it to configure the YOLO-World classes. citeturn4file0

### Visual Question Answering

Use the VQA functionality to ask questions about an uploaded image, such as:

```text
What is the person holding?
How many people are visible?
What color is the car?
Is there a dog in the image?
```

The answer is generated by the BLIP VQA model and can also be converted to speech. citeturn4file0

---

## 🔌 API routes

| Route | Method | Purpose |
|---|---|---|
| `/` | `GET` | Renders the main web interface. |
| `/detect` | `POST` | Accepts uploaded media and runs object detection. |
| `/vqa` | `POST` | Accepts a question and image filename, then returns a VQA answer. |

### `/detect`

Multipart form fields used by the current backend include:

```text
image       Uploaded image/video
objects     Optional comma-separated target classes
confidence  Detection confidence threshold
color       Bounding-box color in #RRGGBB format
```

The response renders the results page with input/output URLs, detection information, and optional audio. citeturn4file0

### `/vqa`

JSON payload:

```json
{
  "question": "What is in the image?",
  "image_filename": "uploaded-image.jpg"
}
```

Successful responses contain the generated answer and, when speech generation succeeds, an audio URL. citeturn4file0

---

## 📸 Example workflow

```text
📤 Upload image
      ↓
🎯 Choose target objects
      ↓
🎚️ Set confidence
      ↓
🔍 Run detection
      ↓
🖼️ View annotated result
      ↓
🔊 Listen to the generated summary
```

For VQA:

```text
🖼️ Select image
      ↓
❓ Ask a question
      ↓
🧠 BLIP VQA inference
      ↓
💬 Receive answer
      ↓
🔊 Optional speech output
```

---

## ⚠️ Current implementation notes

- The backend currently initializes `yolov8s-world.pt` directly, so the weight file needs to be present at runtime. citeturn4file0
- The frontend exposes both **Bounding Box Detection** and **Instance Segmentation** controls, but the current `/detect` backend code shown in the repository performs YOLO detection and does not implement a separate segmentation inference path. citeturn4file0 citeturn6file0
- Video detection summarizes detected class names after processing rather than returning per-frame counts. citeturn4file0
- Uploaded and generated files are scheduled for deletion after approximately 15 minutes. citeturn4file0
- gTTS requires network access when generating speech audio.

These notes are documented intentionally so the README describes the repository as it exists rather than promising functionality that is not currently implemented.

---

## 🛠️ Troubleshooting

### `ModuleNotFoundError`

Make sure the virtual environment is active and dependencies are installed:

```bash
pip install -r requirements.txt
```

### YOLO weight/model error

Verify that `yolov8s-world.pt` is available from the working directory used to launch the application.

### VQA model loading problems

Ensure the machine has internet access the first time the Hugging Face BLIP model is downloaded and that PyTorch/Transformers installed successfully.

### Uploaded image cannot be read

The application first attempts OpenCV image loading and includes a HEIF fallback using `pi-heif` and Pillow. citeturn4file0

### Audio is missing

Speech generation is handled separately from inference. If gTTS fails, the application logs the error and continues without an audio URL. citeturn4file0

---

## 🔐 Security & production considerations

Before exposing this application publicly, consider adding:

- Upload size and file-type validation
- Authentication/rate limiting
- Safer isolation for untrusted uploads
- Explicit resource limits for video processing
- Production WSGI serving instead of Flask's development server
- More robust temporary-file lifecycle management
- Monitoring and structured application logging

---

## 🌱 Future improvements

Potential next steps for the project:

- [ ] Add true instance-segmentation inference
- [ ] Add drag-and-drop uploads
- [ ] Add webcam/live-camera detection
- [ ] Display richer per-class statistics
- [ ] Add downloadable result bundles
- [ ] Add REST API documentation with OpenAPI
- [ ] Add automated tests
- [ ] Add GPU/CPU performance information
- [ ] Add deployment instructions
- [ ] Add sample screenshots and demo media

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature/my-improvement
```

3. Make and test your changes.
4. Commit your work:

```bash
git add .
git commit -m "feat: improve vision workflow"
```

5. Push the branch and open a pull request.

---

## 📄 License

No license file is currently present in the repository. If you plan to distribute or reuse this project, add an appropriate `LICENSE` file and update this section.

---

<div align="center">

### Built with Python, Flask, YOLO-World, BLIP, OpenCV & PyTorch 🧠👁️

**[🌐 Try the Live Website](https://world-object-detector.netlify.app/)** · **[⭐ Star the repository](https://github.com/owaies/ObjectDetector)**

</div>
