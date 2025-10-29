
---

# 🚫 Wrong-Way Traffic Violation Detection using YOLOv8

This project detects vehicles moving in the **wrong direction** using object detection and simple line-crossing logic. It uses the **YOLOv8 object detector** along with a built-in tracker to identify vehicles and determine if they violate the lane direction based on two defined lines in the frame.

---

## 📌 Project Objective

To detect and flag **vehicles** (cars, bikes, trucks, buses, etc.) that cross a set of two defined lines **in the wrong direction** — essentially identifying wrong-way driving violations in a road surveillance video.

---

## 🛠️ Methodology

### 1. **Input Setup**

* Input frames are loaded from:
  `'/content/drive/MyDrive/bike_lane/frames_picoR_crop3'`
* Output frames with violations marked are saved to:
  `'/content/output_frames(1)'`

### 2. **YOLOv8 Detection**

* The model used: `yolov8n.pt` (a small and fast YOLO model).
* YOLO detects objects in each frame (cars, bikes, persons, trucks, etc.).
* Tracked using built-in tracking features (`model.track`).

### 3. **Watched Classes**

The detection is filtered to only care about these classes:

* `0 = person`
* `1 = bicycle`
* `2 = car`
* `3 = motorbike`
* `5 = bus`
* `7 = truck`

### 4. **Line-Based Violation Logic**

* Two lines are defined:

  * **LINE1** (entry point): `((42, 379), (441, 351))`
  * **LINE2** (exit point): `((67, 508), (575, 473))`
* Each object's center is checked:

  * If it **first touches LINE1** and **then touches LINE2**, it's considered a **wrong-way violation**.
  * Once a violation is detected, the object is tracked and marked in **red** with the label `"Violation"`.

### 5. **Output**

* For each frame:

  * Violating objects are highlighted with a red box and label.
  * Saved to the output directory with annotations.

---

## 🧠 How It Works (Simple Terms)

1. We **read each video frame** as an image.
2. YOLO detects **vehicles and persons** and tracks their movement.
3. If a vehicle **enters from the wrong side** (by touching LINE1) and then **crosses the next line** (LINE2), we assume it’s going the **wrong way**.
4. That vehicle is now considered **violating**, and we **draw a red box** and label on it.
5. This is done for every frame, and results are saved.

---

## ✅ How to Modify the Code

### ➤ To Track Only Cars and Bikes:

Change the line:

```python
WATCHED_CLASSES = {0, 1, 2, 3, 5, 7}
```

to:

```python
WATCHED_CLASSES = {2, 3}
```

(2 = car, 3 = motorbike)

---

## 📁 Project Folder Structure 

```
wrong_way_detection/
│
├── result/              # output video
├── wrong_way_detection.pynb         #  detection code
├── README.md                 
```

---

## 🧪 Sample Use-Cases

* Detecting wrong-way drivers on bike lanes
* Flagging vehicles violating one-way traffic rules
* Monitoring restricted entry zones

---

## 🚀 Requirements

* Python
* OpenCV
* [Ultralytics YOLO](https://docs.ultralytics.com)

Install YOLO:

```bash
pip install ultralytics
```

---

## 📞 Contact

For questions or feedback, feel free to reach out!

---
