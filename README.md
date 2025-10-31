# 🚦 Smart Traffic Violation Detection System

> **AI-powered traffic monitoring app built with YOLOv8 + Streamlit + OpenCV to detect, analyze, and report traffic rule violations in real-time.**

---

## 🧠 Overview

In a world where traffic chaos is a daily scene, **Smart Traffic Violation System** is designed to bring automation, accuracy, and accountability to traffic monitoring.
This project uses **Computer Vision** and **Deep Learning** to detect vehicles violating rules like- helmet violation, wrong-lane violation, overspeed violation, red light violation and triple riding violation  — all powered by **YOLOv8 Object Detection** and presented with an interactive **Streamlit frontend**.

💡 The goal?
To **assist traffic authorities** in identifying violators faster, **reduce human error**, and **enhance road safety** using AI-driven surveillance.

---

## ⚙️ Tech Stack

| Layer                     | Tools / Frameworks Used                         |
| ------------------------- | ----------------------------------------------- |
| 🧩 **Frontend**           | Streamlit (Python GUI Framework)                |
| 🧠 **Model / Backend**    | YOLOv8, OpenCV, PyTorch                         |
| 📊 **Data Handling**      | Pandas, NumPy                                   |
| 🗂️ **Storage / Dataset** | Custom Indian traffic violation dataset         |
| 🔍 **Visualization**      | Matplotlib, Seaborn                             |
| 🖥️ **Deployment**        | Render / Hugging Face / Streamlit Community Cloud       |

---

## 📦 Core Modules

### 1️⃣ **Object Detection Module**

* Implemented using **YOLOv8 (Ultralytics)**.
* Detects vehicles, number plates, helmets, and pedestrians.
* Trained with a **custom dataset** (`Indian_Traffic_Violations.csv`) containing labeled images of traffic scenarios.
* Uses **OpenCV** for frame-by-frame analysis.

---

### 2️⃣ **Violation Analysis Module**

* Takes YOLO’s bounding box predictions.
* Applies business logic to identify specific violations (e.g., vehicle crossed stop line, no helmet detected).
* Logs detected violations in a `.csv` file (`detection_log.csv`) for record keeping.

---

### 3️⃣ **Data Analytics Module**

* Uses **Pandas** and **Matplotlib** to visualize daily, weekly, or monthly violations.
* Generates meaningful dashboards like:

  * 🚗 Most frequent violation type
  * 📍 Hotspot zones
  * 🕓 Time-based trends

---

### 4️⃣ **Frontend UI (Streamlit App)**

* Built with **Streamlit** (`app.py` / `main.py`).
* Provides a clean, interactive web dashboard.
* Allows users to:

  * Upload images or video files along with live web cam for detection 🎥
  * View live detection preview
  * Access logged results and analytics 📊

---

## 🎯 Project Objectives

✅ Automate traffic violation detection.
✅ Reduce dependency on manual CCTV monitoring.
✅ Create a scalable, AI-driven model that adapts to real-world road data.
✅ Build a modern, interactive UI that visualizes live detection results.

---

## 🚗 Dataset Details

📁 **Dataset Name:** Indian Traffic Violations
📊 **Format:** CSV + YOLO-format labeled images
🖼️ **Classes:**

1. Car 🚘
2. Bike 🏍️
3. Person 🧍
4. Helmet ⛑️
5. Traffic Light 🚦
6. Number Plate 🔢

The dataset was preprocessed and labeled using **LabelImg**, and large training images and videos were excluded from GitHub due to storage limits.

> ⚠️ To run this project locally, download the required dataset and model weights from the links below (you can update them to your Drive/Hugging Face once uploaded):
>
> * [📥 Dataset (Indian_Traffic_Violations.csv)](#)
> * [📥 YOLOv8 Weights (best.pt)](#)

---

## 🧰 Features

✨ **Real-time Detection** – Capture violations from live video streams or pre-recorded files.
📊 **Analytics Dashboard** – Graphical insights for authorities.
🧠 **AI-Powered YOLOv8 Model** – Highly accurate, lightning-fast predictions.
🧾 **Violation Log File** – Auto-generated CSV with timestamp and details.
🎨 **Interactive UI** – Simple, modern interface built with Streamlit.
🌐 **Deployable Anywhere** – Supports deployment on Hugging Face, Render, or Streamlit Community Cloud.

---

## 🧩 How It Works

1️⃣ The user uploads a video or image file via Streamlit interface.
2️⃣ The YOLOv8 model processes the input frame-by-frame using OpenCV.
3️⃣ The system identifies vehicles, helmets, and traffic lights.
4️⃣ If a rule is violated (like no helmet / crossing red light), it logs the violation with time, frame ID, and detected object type.
5️⃣ Data is stored in `detection_log.csv` and displayed as live visuals on the dashboard.

---

## 🛠️ Setup Instructions

To run this project locally:

### Step 1: Clone the Repo

```bash
git clone https://github.com/suraktimssm/Smart-Traffic-Violation-System.git
cd Smart-Traffic-Violation-System
```

### Step 2: Install Dependencies

Make sure you have Python 3.10+ installed.
Then run:

```bash
pip install -r requirements.txt
```

> **requirements.txt** includes:

```
opencv-python-headless
ultralytics
easyocr
pandas
numpy
plotly
streamlit
requests

```

---

### Step 3: Download Required Files

Since large files were removed to meet GitHub storage limits, you’ll need to manually download these:

| File / Folder                   | Description                            | Action                                     |
| ------------------------------- | -------------------------------------- | ------------------------------------------ |
| `best.pt`                       | Trained YOLOv8 model weights           | [Download here](#) and place in `/weights` |
| `Indian_Traffic_Violations.csv` | Dataset for testing & analytics        | [Download here](#) and place in `/data`    |
| `/train` & `/valid` folders     | Labeled images for training/validation | Optional (for retraining YOLO model)       |

---

### Step 4: Run the App

```bash
streamlit run app.py
```

or

```bash
streamlit run main.py
```

---

## 💡 Challenges & Strategies

| Challenge                        | Strategy Adopted                                                |
| -------------------------------- | --------------------------------------------------------------- |
| ⚙️ Large dataset size            | Split and cleaned dataset, used `.gitignore` for heavy assets   |
| 🎥 Processing lag for HD videos  | Used frame skipping and optimized inference                     |
| 🧠 YOLO model misclassifications | Fine-tuned weights on Indian traffic dataset                    |
| 🗃️ GitHub 100 MB limit          | Offloaded large files to external storage                       |
| 💻 Deployment issues             | Streamlined `requirements.txt` & fixed Python version conflicts |

---

## 🚀 Future Expansion

🛰️ **Integration with IoT CCTV feeds** for real-time monitoring.
📱 **Mobile App Dashboard** for on-the-go access.
🧾 **Auto-generated challans** linked with vehicle registration numbers.
📊 **AI-powered behavior analysis** – speed, aggression, and traffic density prediction.
🌍 **Multi-city deployment support** with cloud synchronization.

---

## 📘 Project Structure

```
📂 Smart-Traffic-Violation-System/
 ┣ 📁 data/                                      # Dataset files (CSV, images)
 ┣ 📁 train/, valid/, test/                      # YOLO labeled data (removed for size)
 ┣ 📁 runs/                                      # Training results (removed for size)
 ┣ 📜 app.py / main.py /                         # Streamlit frontend
 ┣ 📜 detector.py                                # YOLOv8 model detection logic
 ┣ 📜 model_training.py                          # Model training script
 ┣ 📜 data_analysis.py                           # Analytics and visualization
 ┣ 📜 requirements.txt                           # Dependencies list
 ┣ 📜 detection_log.csv                          # Logs of detected violations
 ┣ 📜 Procfile                                   # For Render/Vercel/Hugging Face deployment
 ┣ 📜 LICENSE                      # MIT License
 ┗ 📜 README.md                    # You’re reading it 😎
```

---

## 💬 Author

👨‍💻 **Suraktim Choudhury**
🎓 B.Tech CSE (4th Year), University of Engineering and Management, Jaipur
🌐 [LinkedIn](https://www.linkedin.com/in/suraktimchoudhury)
💼 Aspiring Blockchain & Python Developer | Using AI trends

---

## 🏁 Conclusion

🚦 The **Smart Traffic Violation Detection System** is a powerful step toward **AI-driven urban management**.
It showcases the perfect blend of **Computer Vision, Automation, and Data Analytics** to build smarter cities — one frame at a time.
Whether you’re a researcher, developer, or traffic department official, this system is your first step towards **next-gen intelligent surveillance**.

✨ *Built with passion, precision, and Python.* 🐍

---
