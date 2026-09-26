# opensource1

# Face Recognition & Mood Detection

A real-time webcam application that detects faces, recognizes registered people, analyzes facial mood from landmarks, estimates age, and generates a short AI comment about the detected mood using a local Gemma model via Ollama.

## Features

- **Face Registration** — Capture and store face images for a new person via webcam.
- **Face Recognition** — Train an LBPH recognizer (OpenCV) on registered faces and identify people in real time.
- **Mood Detection** — Uses MediaPipe Face Mesh landmarks to classify mood as Happy, Sad, Surprised, Frustrated, Focused, or Neutral.
- **Age Estimation** — Optional Caffe-based age classifier (bucketed ranges like `(25-32)`).
- **Local AI Commentary** — Sends the detected mood to a local [Ollama](https://ollama.ai/) instance running `gemma2:2b` and displays a short, friendly response on-screen.
- **Mood History Logging** — Stores a per-person mood history (timestamp, mood, confidence, age) in a local JSON file, viewable from the CLI menu.

## Requirements

- Python 3.8+
- A webcam
- [Ollama](https://ollama.ai/) installed and running locally with the `gemma2:2b` model pulled (optional — the app runs without it, just without AI commentary)

### Python dependencies

```bash
pip install opencv-contrib-python mediapipe numpy requests
```

> **Note:** `opencv-contrib-python` (not plain `opencv-python`) is required, since `cv2.face.LBPHFaceRecognizer_create()` lives in the `contrib` face module.

### Optional: Age estimation model files

To enable age estimation, download the following Caffe model files and place them in the `models/` directory:

- `age_deploy.prototxt`
- `age_net.caffemodel`

These are commonly available from public age/gender estimation model repositories (e.g. the Adience/CVCL age-gender Caffe models). If the files are missing, the app runs normally with age display disabled.

## Project Structure

```
.
├── main.py              # (this script)
├── dataset/             # Captured face images, one folder per registered person
├── models/
│   ├── face_recognizer.yml
│   ├── labels.json
│   ├── mood_history.json
│   ├── age_deploy.prototxt   (optional, user-supplied)
│   └── age_net.caffemodel    (optional, user-supplied)
```

These folders are created automatically at runtime — no manual setup required.

## Usage

Run the script and choose an option from the menu:

```bash
python main.py
```

```
1. Register a new face
2. Train recognizer
3. Run live app
4. View mood history
```

1. **Register a new face** — Prompts for consent, then captures ~25 face images via webcam for a named person, saved under `dataset/<name>/`.
2. **Train recognizer** — Trains the LBPH face recognizer on all images in `dataset/` and saves the model to `models/`.
3. **Run live app** — Opens the webcam feed with real-time face recognition, mood detection, age estimation, and AI commentary overlaid on screen. Press `q` or `ESC` to quit.
4. **View mood history** — Lists all people with recorded mood history and lets you view recent entries for one person or all of them.

## Configuration

Key parameters can be adjusted near the top of the script:

| Variable | Description |
|---|---|
| `OLLAMA_URL` | Ollama API endpoint |
| `MODEL_NAME` | Ollama model used for commentary (default: `gemma2:2b`) |
| `REQUEST_COOLDOWN` | Minimum seconds between Gemma requests |
| `IMAGES_PER_PERSON` | Number of images captured per registration |
| `RECOGNITION_CONFIDENCE_THRESHOLD` | LBPH distance threshold for a positive match (lower = stricter) |
| `MOOD_ANALYSIS_INTERVAL` | Seconds between mood re-analysis |
| `AGE_ANALYSIS_INTERVAL` | Seconds between age re-analysis |
| `GEMMA_ONLY_ON_MOOD_CHANGE` | If `True`, only queries Gemma when mood changes |

## Privacy Note

This application captures and stores facial images and mood history **locally only** — no data is sent anywhere except to a local Ollama instance for generating short text responses. Registration explicitly asks for consent before capturing any images.

## License

Add a license of your choice (e.g. MIT) here.
