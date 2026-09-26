import os
import json
import time
import math
import cv2
import numpy as np
import mediapipe as mp
import requests
from datetime import datetime

# --- Config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"
REQUEST_COOLDOWN = 5.0

DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "face_recognizer.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
HISTORY_PATH = os.path.join(MODEL_DIR, "mood_history.json")

# Age model files - download separately, see note above.
AGE_PROTO_PATH = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL_PATH = os.path.join(MODEL_DIR, "age_net.caffemodel")
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

IMAGES_PER_PERSON = 25
MAX_HISTORY_PER_PERSON = 200

RECOGNITION_CONFIDENCE_THRESHOLD = 70
MOOD_ANALYSIS_INTERVAL = 1.0
AGE_ANALYSIS_INTERVAL = 2.0  # age changes slowly, so check it less often
GEMMA_ONLY_ON_MOOD_CHANGE = True

_last_request_time = 0.0
_last_response = ""


# --- Gemma / Ollama ---
def ask_gemma(mood, person_name="User"):
    global _last_request_time, _last_response

    now = time.time()
    if now - _last_request_time < REQUEST_COOLDOWN:
        return _last_response

    prompt = (
        f"A person appears to be feeling '{mood}'. "
        "In one short, friendly sentence, give a brief observation or "
        "gentle suggestion appropriate for someone feeling this way. "
        "Do not mention that you are an AI or reference any image or camera."
    )

    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=8)
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        _last_response = text if text else "(No response from Gemma.)"
    except requests.exceptions.ConnectionError:
        _last_response = "[Ollama not reachable - is 'ollama serve' running?]"
    except requests.exceptions.Timeout:
        _last_response = "[Ollama request timed out.]"
    except Exception as e:
        _last_response = f"[Gemma error: {e}]"

    _last_request_time = now
    return _last_response


# --- Age Estimation ---
def load_age_model():
    """
    Loads the Caffe age-estimation model if both files are present.
    Returns None if unavailable, so the rest of the app can run without it.
    """
    if not os.path.exists(AGE_PROTO_PATH) or not os.path.exists(AGE_MODEL_PATH):
        print("Warning: Age model files not found in 'models/'. Age display disabled.")
        print(f"Expected: {AGE_PROTO_PATH} and {AGE_MODEL_PATH}")
        return None

    try:
        net = cv2.dnn.readNetFromCaffe(AGE_PROTO_PATH, AGE_MODEL_PATH)
        return net
    except Exception as e:
        print(f"Warning: Could not load age model ({e}). Age display disabled.")
        return None


def estimate_age(age_net, frame, box):
    """Runs the age-classification model on the cropped face. Returns a bucket string like '(25-32)'."""
    if age_net is None:
        return None

    x, y, w, h = box
    face_crop = frame[max(0, y):y + h, max(0, x):x + w]
    if face_crop.size == 0:
        return None

    try:
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), AGE_MODEL_MEAN, swapRB=False)
        age_net.setInput(blob)
        preds = age_net.forward()
        return AGE_BUCKETS[preds[0].argmax()]
    except Exception as e:
        print(f"Age estimation error: {e}")
        return None


# --- Mood History Storage ---
def load_history():
    if not os.path.exists(HISTORY_PATH):
        return {}
    try:
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print("Warning: history file was unreadable, starting a fresh one.")
        return {}


def save_history(history):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def log_mood(history, person_name, mood, confidence, age_bucket):
    if person_name == "Unknown Person":
        return history

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mood": mood,
        "confidence": round(confidence, 1),
        "age_estimate": age_bucket if age_bucket else "N/A"
    }

    person_log = history.get(person_name, [])
    person_log.append(entry)
    if len(person_log) > MAX_HISTORY_PER_PERSON:
        person_log = person_log[-MAX_HISTORY_PER_PERSON:]

    history[person_name] = person_log
    save_history(history)
    return history


def view_history():
    history = load_history()
    if not history:
        print("No mood history recorded yet.")
        return

    print("\nRegistered people with history:", list(history.keys()))
    name = input("Enter a name to view their history (blank = all): ").strip()

    people_to_show = {name: history[name]} if name and name in history else history
    if name and name not in history:
        print(f"No history found for '{name}'.")
        return

    for person, entries in people_to_show.items():
        print(f"\n--- {person} ({len(entries)} entries) ---")
        for e in entries[-20:]:
            age_str = e.get("age_estimate", "N/A")
            print(f"  {e['timestamp']}  |  {e['mood']}  ({e['confidence']}%)  |  Age: {age_str}")


# --- Registration ---
def get_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Could not load Haar Cascade file - check your OpenCV install.")
    return detector


def register_new_face():
    print("This will capture your face using the webcam and store the images")
    print("locally in the 'dataset' folder, for face recognition purposes only.")
    consent = input("Do you consent to registering your face? (yes/no): ").strip().lower()
    if consent not in ("yes", "y"):
        print("Registration cancelled.")
        return

    name = input("Enter the person's name: ").strip()
    if not name:
        print("Name cannot be empty. Aborting.")
        return

    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    try:
        detector = get_face_detector()
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print(f"Capturing {IMAGES_PER_PERSON} images. Move your head slightly. Press 'q'/ESC to stop early.")

    count = 0
    while cap.isOpened() and count < IMAGES_PER_PERSON:
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_crop = gray[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (200, 200))
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_crop)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {count}/{IMAGES_PER_PERSON}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break

        cv2.imshow("Registering Face - Press q/ESC to stop", frame)
        key = cv2.waitKey(200) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"Done. Saved {count} images for '{name}'. Now run training.")
    else:
        print("No face images were captured.")


# --- Training ---
def train_recognizer():
    if not os.path.isdir(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print("No registered faces found. Register a face first.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, labels, label_map = [], [], {}
    next_id = 0

    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[next_id] = person_name
        count = 0
        for img_file in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(next_id)
            count += 1
        print(f"  Loaded {count} images for '{person_name}'")
        next_id += 1

    if not faces:
        print("No usable images found.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f)

    print(f"Training complete. Registered people: {list(label_map.values())}")


# --- Recognition ---
def load_recognizer():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("No trained recognizer found. All faces will show as 'Unknown Person'.")
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        raw_map = json.load(f)
    return recognizer, {int(k): v for k, v in raw_map.items()}


def recognize_face(recognizer, label_map, gray_frame, box):
    if recognizer is None:
        return "Unknown Person", None

    x, y, w, h = box
    face_crop = gray_frame[y:y + h, x:x + w]
    if face_crop.size == 0:
        return "Unknown Person", None

    face_crop = cv2.resize(face_crop, (200, 200))
    label_id, distance = recognizer.predict(face_crop)

    if distance <= RECOGNITION_CONFIDENCE_THRESHOLD:
        name = label_map.get(label_id, "Unknown Person")
    else:
        name = "Unknown Person"

    return name, distance


# --- Landmark-based Mood Detection ---
def _dist(p1, p2, w, h):
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)


def analyze_mood_from_landmarks(landmarks, w, h):
    lm = landmarks.landmark

    top_face = lm[10]
    bottom_face = lm[152]
    left_mouth = lm[61]
    right_mouth = lm[291]
    top_lip = lm[13]
    bottom_lip = lm[14]
    left_eyebrow = lm[105]
    right_eyebrow = lm[334]
    left_eye_top = lm[159]
    left_eye_bottom = lm[145]
    right_eye_top = lm[386]
    right_eye_bottom = lm[374]

    face_height = _dist(top_face, bottom_face, w, h)
    if face_height == 0:
        return "Neutral", 50

    mouth_open_ratio = _dist(top_lip, bottom_lip, w, h) / face_height

    mouth_center_y = (top_lip.y + bottom_lip.y) / 2
    corner_avg_y = (left_mouth.y + right_mouth.y) / 2
    smile_ratio = (mouth_center_y - corner_avg_y) * h / face_height

    brow_eye_dist = (
        _dist(left_eyebrow, left_eye_top, w, h) + _dist(right_eyebrow, right_eye_top, w, h)
    ) / (2 * face_height)

    eye_open = (
        _dist(left_eye_top, left_eye_bottom, w, h) + _dist(right_eye_top, right_eye_bottom, w, h)
    ) / (2 * face_height)

    if mouth_open_ratio > 0.06 and eye_open > 0.028:
        mood, confidence = "Surprised", 75
    elif smile_ratio > 0.010:
        mood, confidence = "Happy", 80
    elif smile_ratio < -0.010 and brow_eye_dist < 0.048:
        mood, confidence = "Sad", 70
    elif brow_eye_dist < 0.040 and mouth_open_ratio < 0.02:
        mood, confidence = "Frustrated", 65
    elif eye_open < 0.024 and brow_eye_dist < 0.052 and abs(smile_ratio) < 0.008:
        mood, confidence = "Focused", 65
    else:
        mood, confidence = "Neutral", 60

    return mood, confidence


def draw_overlay(frame, box, name, mood, mood_conf, age_bucket, gemma_text):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = f"{name}  Age: {age_bucket}" if age_bucket else name
    cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if mood:
        cv2.putText(frame, f"{mood} ({mood_conf:.0f}%)", (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    if gemma_text:
        max_chars = 60
        lines = [gemma_text[i:i + max_chars] for i in range(0, len(gemma_text), max_chars)]
        for i, line in enumerate(lines[:2]):
            cv2.putText(frame, line, (10, frame.shape[0] - 40 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def run_main_app():
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    recognizer, label_map = load_recognizer()
    age_net = load_age_model()
    history = load_history()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    last_mood_time = 0.0
    last_age_time = 0.0
    last_mood = None
    last_mood_conf = 0.0
    last_age = None
    last_gemma_text = ""

    print("Starting Face Recognition and Mood Detection. Press 'q' or ESC to quit.")

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection, \
         mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detection_result = face_detection.process(rgb_frame)

            if detection_result.detections:
                detection = detection_result.detections[0]
                rel_box = detection.location_data.relative_bounding_box
                x = max(0, int(rel_box.xmin * w))
                y = max(0, int(rel_box.ymin * h))
                bw = int(rel_box.width * w)
                bh = int(rel_box.height * h)
                box = (x, y, bw, bh)

                name, distance = recognize_face(recognizer, label_map, gray_frame, box)

                mesh_result = face_mesh.process(rgb_frame)
                face_landmarks = None
                if mesh_result.multi_face_landmarks:
                    face_landmarks = mesh_result.multi_face_landmarks[0]
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )

                now = time.time()

                # Age changes slowly, so refresh less often than mood
                if now - last_age_time >= AGE_ANALYSIS_INTERVAL:
                    age_result = estimate_age(age_net, frame, box)
                    if age_result:
                        last_age = age_result
                    last_age_time = now

                if face_landmarks and now - last_mood_time >= MOOD_ANALYSIS_INTERVAL:
                    mood, mood_conf = analyze_mood_from_landmarks(face_landmarks, w, h)
                    last_mood_time = now
                    mood_changed = (mood != last_mood)
                    last_mood, last_mood_conf = mood, mood_conf

                    history = log_mood(history, name, mood, mood_conf, last_age)

                    if not GEMMA_ONLY_ON_MOOD_CHANGE or mood_changed:
                        last_gemma_text = ask_gemma(mood, name)

                draw_overlay(frame, box, name, last_mood, last_mood_conf, last_age, last_gemma_text)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Face Recognition and Mood Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# --- Entry Point ---
if __name__ == "__main__":
    while True:
        print("1. Register a new face")
        print("2. Train recognizer")
        print("3. Run live app")
        print("4. View mood history")
        choice = input("Choose an option (1/2/3/4): ").strip()

        if choice == "1":
            register_new_face()
            break
        elif choice == "2":
            train_recognizer()
            break
        elif choice == "3":
            run_main_app()
            break
        elif choice == "4":
            view_history()
            break
        else:
            print(f"Invalid choice: '{choice}'. Please type 1, 2, 3, or 4.\n")
