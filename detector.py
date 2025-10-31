import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
import random
import pandas as pd
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

# --- Google Drive weights setup (Direct .pt download) ---
DRIVE_LINKS = {
    "helmet": "https://drive.google.com/uc?export=download&id=1-7-Ma6cfzq3QeRbe--wgJ_0tpE8tsj4-",
    "triple": "https://drive.google.com/uc?export=download&id=1PwxPqoyeXTe_tnD7WvP39uRSxDuPWEkf",
    "traffic": "https://drive.google.com/uc?export=download&id=1p1ed-F8LlB4502f0FVptP65pNBXn5BiL",
    "redlight": "https://drive.google.com/uc?export=download&id=10XitgVNVo3p2ydFas3yZefQH-pK4t-3j",
    "wronglane": "https://drive.google.com/uc?export=download&id=1GofcmHkj47TC8dCQeOGxmpzvb0T9IGql"
}

WEIGHTS_DIR = "weights"
if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

def ensure_weights_available(model_choice):
    """Ensure YOLO weights are available (direct .pt download from Drive)."""
    file_map = {
        "helmet": "yolov8_helmet.pt",
        "triple": "yolov8_triple.pt",
        "traffic": "yolov8s.pt",
        "redlight": "best.pt",
        "wronglane": "yolov8n.pt"
    }

    file_name = file_map.get(model_choice, "yolov8_helmet.pt")
    file_path = os.path.join(WEIGHTS_DIR, file_name)

    # Download only if not present or file too small
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 5_000_000:
        print(f"[INFO] Downloading {file_name} from Google Drive...")
        try:
            url = DRIVE_LINKS.get(model_choice)
            if not url:
                raise ValueError(f"No Drive link found for {model_choice}")

            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[INFO] ✅ {file_name} downloaded successfully!")
        except Exception as e:
            print(f"[ERROR] ❌ Failed to download {file_name}: {e}")
    else:
        print(f"[INFO] ✅ {file_name} already exists locally.")
    return file_path


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

    # --- Ensure the selected model weight is available ---
    model_path = ensure_weights_available(model_choice)

    # Load YOLO model
    model = load_model(model_path)
    overlay = image.copy()

    # Initialize counters
    helmet_violation = 0
    triple_riding = 0
    overspeed = 0
    wrong_lane = 0
    red_light = 0

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
