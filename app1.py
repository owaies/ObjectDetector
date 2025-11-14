import os
import uuid
import cv2
import time
import threading
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from collections import defaultdict
from gtts import gTTS
from playsound import playsound

# --- Initialize App and Configure Folders ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'
AUDIO_FOLDER = 'static/audio/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
# The following lines are now handled within the cleanup scheduler
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# os.makedirs(AUDIO_FOLDER, exist_ok=True)

# --- Load Model Once ---
print("Loading YOLO-World model...")
model = YOLO('yolov8s-world.pt')
print("Model loaded successfully.")


# --- NEW: Background File Cleanup Scheduler ---
def schedule_cleanup(files_to_delete, delay_seconds=900): # 900 seconds = 15 minutes
    """
    Deletes a list of files after a specified delay.
    This runs in a separate thread so it doesn't block the web request.
    """
    def cleanup_task():
        print(f"SCHEDULER: Waiting {delay_seconds} seconds to clean up {len(files_to_delete)} files.")
        time.sleep(delay_seconds)
        
        deleted_folders = set()
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"CLEANUP: Deleted file {file_path}")
                    
                    # Try to delete the parent folder if it's empty
                    folder = os.path.dirname(file_path)
                    if folder not in deleted_folders and not os.listdir(folder):
                        os.rmdir(folder)
                        print(f"CLEANUP: Deleted empty directory {folder}")
                        deleted_folders.add(folder)
                else:
                    print(f"CLEANUP: File not found, already deleted: {file_path}")
            except Exception as e:
                print(f"Error during cleanup of {file_path}: {e}")
        print("SCHEDULER: Cleanup task finished.")

    # A daemon thread will exit when the main program exits.
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
# --------------------------------------------

# --- Main Page Route ---
@app.route('/')
def index():
    # Ensure directories exist for the session
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    return render_template('index.html')

# --- Detection Route ---
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No image part", 400
    file = request.files['image']
    if file.filename == '':
        return "No selected image", 400
    
    user_input = request.form.get('objects', '').lower().strip()
    if not user_input:
        return "No object names entered", 400
    
    target_objects = [item.strip() for item in user_input.split(',')]

    if file:
        unique_id = uuid.uuid4().hex
        input_filename = f"{unique_id}_{secure_filename(file.filename)}"
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_filepath)

        model.set_classes(target_objects)
        results = model(input_filepath)
        result = results[0]

        image = cv2.imread(input_filepath)
        detection_counts = defaultdict(int)

        if len(result.boxes) > 0:
            for box in result.boxes:
                if box.conf[0] > 0.25:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    label = f"{class_name}: {confidence:.2f}"
                    detection_counts[class_name.lower()] += 1
                    
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(image, (xmin, ymin), (xmin + label_size[0], ymin + label_size[1] + 10), (0,0,0), -1)
                    cv2.putText(image, label, (xmin + 5, ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        output_filename = f"{unique_id}_output.jpg"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        cv2.imwrite(output_filepath, image)

        audio_url = None
        audio_filepath = None
        if detection_counts:
            announcements = []
            for obj_name, count in detection_counts.items():
                plural_name = obj_name if count == 1 else ('people' if obj_name == 'person' else obj_name + 's')
                announcements.append(f"{count} {plural_name}")
            final_announcement = "I found " + ", and ".join(announcements)
        else:
            final_announcement = "I could not find any of the requested objects."
        
        try:
            tts = gTTS(text=final_announcement, lang='en')
            audio_filename = f"{unique_id}_speech.mp3"
            audio_filepath = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
            tts.save(audio_filepath)
            audio_url = url_for('static', filename=f'audio/{audio_filename}')
        except Exception as e:
            print(f"Error with gTTS: {e}")
        
        # --- NEW: Schedule all generated files for deletion ---
        files_to_delete = [input_filepath, output_filepath]
        if audio_filepath:
            files_to_delete.append(audio_filepath)
        schedule_cleanup(files_to_delete)
        # ----------------------------------------------------

        input_image_url = url_for('static', filename=f'uploads/{input_filename}')
        output_image_url = url_for('static', filename=f'outputs/{output_filename}')

        return render_template('results.html', 
                               input_image_url=input_image_url, 
                               output_image_url=output_image_url,
                               detection_summary=dict(detection_counts),
                               audio_url=audio_url)

    return "Error processing file", 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
