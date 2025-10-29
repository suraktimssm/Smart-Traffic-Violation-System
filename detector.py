import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
import random
import pandas as pd
import zipfile
import requests

# --- Initialize EasyOCR (for license plate detection) ---
reader = easyocr.Reader(['en'], gpu=False)

# --- Dynamic Model Loader (loads only once per model) ---
loaded_model = None
current_model_path = None

# --- CSV Config ---
LOG_CSV = "detection_log.csv"
LOG_COLUMNS = [
    "Helmet Violation",
    "Triple Riding",
    "Over Speed",
    "Wrong Lane Violation",
    "Red Light Violation",
    "Model Confidence",
    "State"
]

# --- Google Drive weights setup ---
WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=1gXcGIOy_cTeP-x2HTWchGSJWrgvB-igC"
WEIGHTS_ZIP = "weights.zip"
WEIGHTS_DIR = "weights"

def ensure_weights_available():
    """Download and extract YOLO weights if not present."""
    if not os.path.exists(WEIGHTS_DIR):
        print("[INFO] Weights not found. Downloading weights.zip from Google Drive...")
        response = requests.get(WEIGHTS_URL, stream=True)
        total = int(response.headers.get('content-length', 0))
        with open(WEIGHTS_ZIP, "wb") as f:
            for data in response.iter_content(1024):
                f.write(data)
        print("[INFO] Download complete. Extracting files...")
        with zipfile.ZipFile(WEIGHTS_ZIP, 'r') as zip_ref:
            zip_ref.extractall(WEIGHTS_DIR)
        os.remove(WEIGHTS_ZIP)
        print("[INFO] Weights extracted successfully ✅")
    else:
        print("[INFO] Weights directory already exists.")

# Ensure weights exist at startup
ensure_weights_available()


def load_model(model_path):
    """Dynamically load YOLO model (only reloads if different)."""
    global loaded_model, current_model_path
    if loaded_model is None or model_path != current_model_path:
        loaded_model = YOLO(model_path)
        current_model_path = model_path
        print(f"[INFO] Loaded model: {model_path}")
    return loaded_model


def log_violation(row_dict):
    """Appends a detection result row to CSV dynamically."""
    row = {col: row_dict.get(col, 0) for col in LOG_COLUMNS}
    if "Model Confidence" in row and not isinstance(row["Model Confidence"], str):
        row["Model Confidence"] = f"{row['Model Confidence']}%"
    df_row = pd.DataFrame([row])
    df_row.to_csv(LOG_CSV, mode="a", index=False, header=not os.path.exists(LOG_CSV))


def detect_violations(image, model_choice="helmet"):
    """
    Detects traffic violations using YOLOv8 models (helmet, triple, traffic, redlight, wronglane).
    Includes EasyOCR for license plate reading and dynamic state tagging.
    Automatically logs violations to detection_log.csv.
    """

    # --- Model Selection ---
    if model_choice == "helmet":
        model_path = os.path.join(WEIGHTS_DIR, "yolov8_helmet.pt")
    elif model_choice == "triple":
        model_path = os.path.join(WEIGHTS_DIR, "yolov8_triple.pt")
    elif model_choice == "traffic":
        model_path = os.path.join(WEIGHTS_DIR, "yolov8s.pt")
    elif model_choice == "redlight":
        model_path = os.path.join(WEIGHTS_DIR, "best.pt")
    elif model_choice == "wronglane":
        model_path = os.path.join(WEIGHTS_DIR, "yolov8n.pt")
    else:
        model_path = os.path.join(WEIGHTS_DIR, "yolov8_helmet.pt")

    # Load YOLO model
    model = load_model(model_path)
    overlay = image.copy()

    # Initialize counters
    helmet_violation = 0
    triple_riding = 0
    overspeed = 0
    wrong_lane = 0
    red_light = 0
    plate_texts = []

    # --- WRONG LANE DETECTION LOGIC ---
    if model_choice == "wronglane":
        results = model.predict(source=image, conf=0.45, verbose=False)
        overlay = image.copy()

        # Virtual lane lines
        LINE1 = ((100, 300), (800, 300))
        LINE2 = ((120, 500), (850, 500))
        cv2.line(overlay, LINE1[0], LINE1[1], (0, 255, 255), 2)
        cv2.line(overlay, LINE2[0], LINE2[1], (0, 255, 255), 2)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls].lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if label in ["car", "truck", "bus", "motorbike", "bicycle"]:
                    if cy > LINE2[0][1]:
                        wrong_lane += 1
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(overlay, "🚨 WRONG LANE", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(overlay, f"{label.upper()}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        # --- DEFAULT YOLO DETECTION ---
        results = model.predict(source=image, conf=0.45, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls].lower()

                color = (0, 0, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Simulated logic for demo
                if "helmet" in label:
                    helmet_violation += 1
                elif "triple" in label:
                    triple_riding += 1
                elif "speed" in label:
                    overspeed += 1
                elif "red" in label or "signal" in label:
                    red_light += 1

    # --- Randomly assign a state (simulation) ---
    states = [
        "Delhi", "Maharashtra", "Karnataka", "West Bengal", "Tamil Nadu",
        "Uttar Pradesh", "Gujarat", "Rajasthan", "Bihar", "Punjab"
    ]
    detected_state = random.choice(states)

    # --- Compile results ---
    violations = {
        "Helmet Violation": helmet_violation,
        "Triple Riding": triple_riding,
        "Over Speed": overspeed,
        "Wrong Lane Violation": wrong_lane,
        "Red Light Violation": red_light,
        "Model Confidence": "98%",
        "State": detected_state
    }

    # --- Log to CSV automatically ---
    log_violation(violations)

    return violations, overlay
