Import os
import uuid
import cv2
import time
import threading
import numpy as np
from flask import Flask, request, render_template, url_for, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import defaultdict
from gtts import gTTS
import pi_heif
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# --- Initialize App and Configure Folders ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'
AUDIO_FOLDER = 'static/audio/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# --- Load AI Models Once at Startup ---
print("Loading AI models...")
detection_model = YOLO('yolov8s-world.pt')
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
print("All models loaded successfully.")


def hex_to_bgr(hex_color):
    """Converts a hex color string (#RRGGBB) to a BGR tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

def schedule_cleanup(paths, delay=900):
    """Deletes files in a background thread after 15 minutes."""
    def cleanup():
        time.sleep(delay)
        for path in paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"Error cleaning up file {path}: {e}")
    threading.Thread(target=cleanup).start()

def is_video_file(filename):
    """Checks if a filename has a common video extension."""
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files: return "No file part", 400
        file = request.files['image']
        if file.filename == '': return "No selected file", 400

        user_input = request.form.get('objects', '').lower().strip()
        confidence_threshold = float(request.form.get('confidence', 0.25))
        box_color_hex = request.form.get('color', '#FFFF00')
        box_color_bgr = hex_to_bgr(box_color_hex)
        
        if user_input:
            target_objects = [item.strip() for item in user_input.split(',')]
            detection_model.set_classes(target_objects)

        unique_id = uuid.uuid4().hex
        original_filename = secure_filename(file.filename)
        input_filename = f"{unique_id}_{original_filename}"
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_filepath)

        detection_counts = defaultdict(int)
        is_video = is_video_file(original_filename)
        output_filename = f"{unique_id}_output.mp4" if is_video else f"{unique_id}_output.jpg"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        if is_video:
            cap = cv2.VideoCapture(input_filepath)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))
            if not out.isOpened(): return "Error: Could not initialize video writer.", 500
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                results = detection_model(frame, stream=True, conf=confidence_threshold)
                for r in results:
                    frame = r.plot(colors=[box_color_bgr] * 80, labels=True)
                    if r.boxes:
                        for box in r.boxes:
                            detection_counts[detection_model.names[int(box.cls[0])].lower()] += 1
                out.write(frame)
            cap.release(); out.release()
            detection_counts = {k: "found" for k in detection_counts.keys()}
        else:
            image = cv2.imread(input_filepath)
            if image is None:
                try:
                    heif = pi_heif.read_heif(input_filepath)
                    image_pil = Image.frombytes(heif.mode, heif.size, heif.data, "raw")
                    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
                except Exception as e: return f"Could not read image: {e}", 400
            results = detection_model(image, conf=confidence_threshold)
            output_image = image.copy()
            for box in results[0].boxes:
                class_name = detection_model.names[int(box.cls[0])]
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                label = f"{class_name}: {box.conf[0]:.2f}"
                detection_counts[class_name.lower()] += 1
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), box_color_bgr, 2)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(output_image, (xmin, ymin), (xmin + label_size[0] + 10, ymin + label_size[1] + 15), (0,0,0), -1)
                cv2.putText(output_image, label, (xmin + 5, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_bgr, 2)
            cv2.imwrite(output_filepath, output_image)
        
        final_announcement = "Processing complete."
        if not is_video:
            if detection_counts:
                # --- THIS IS THE CORRECTED LINE ---
                announcements = [f"{c} {('people' if n == 'person' else n + 's') if c > 1 else n}" for n, c in detection_counts.items()]
                final_announcement = "I found " + ", and ".join(announcements)
            else: final_announcement = "I could not find any of the requested objects."
        
        audio_url = None
        audio_filepath = ''
        try:
            tts = gTTS(text=final_announcement, lang='en')
            audio_filename = f"{unique_id}_speech.mp3"
            audio_filepath = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            tts.save(audio_filepath)
            audio_url = url_for('static', filename=f'audio/{audio_filename}')
        except Exception as e: print(f"Error with gTTS: {e}")

        schedule_cleanup([input_filepath, output_filepath, audio_filepath])

        return render_template('results.html', 
                               is_video=is_video,
                               input_url=url_for('static', filename=f'uploads/{input_filename}'), 
                               input_filename=input_filename,
                               output_url=url_for('static', filename=f'outputs/{output_filename}'),
                               detection_summary=dict(detection_counts),
                               audio_url=audio_url)
    except Exception as e:
        import traceback
        print(f"!!!!!!!! An error occurred in the main detect route: !!!!!!!!\n{traceback.format_exc()}")
        return "An internal error occurred.", 500

@app.route('/vqa', methods=['POST'])
def vqa():
    data = request.get_json()
    question = data.get('question')
    input_filename = data.get('image_filename')
    unique_id = uuid.uuid4().hex

    if not question or not input_filename:
        return jsonify({'error': 'Missing question or image filename.'}), 400

    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        raw_image = Image.open(image_path).convert('RGB')

        inputs = vqa_processor(raw_image, question, return_tensors="pt")
        out = vqa_model.generate(**inputs)
        answer = vqa_processor.decode(out[0], skip_special_tokens=True)

        audio_url = None
        audio_filepath = ''
        try:
            tts = gTTS(text=answer, lang='en')
            audio_filename = f"{unique_id}_vqa_speech.mp3"
            audio_filepath = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            tts.save(audio_filepath)
            audio_url = url_for('static', filename=f'audio/{audio_filename}')
        except Exception as e:
            print(f"Error generating VQA audio: {e}")

        schedule_cleanup([audio_filepath])
        return jsonify({'answer': answer, 'audio_url': audio_url})

    except Exception as e:
        print(f"!!!!!!!! VQA Error: {e} !!!!!!!!")
        return jsonify({'error': 'Failed to answer the question.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
