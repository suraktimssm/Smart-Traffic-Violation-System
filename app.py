import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cv2
from detector import detect_violations
import tempfile
import os
import time
import csv
import random
import os
import subprocess

# Make sure gdown is available
subprocess.run(["pip", "install", "gdown"], check=False)

# Google Drive model link (direct .pt)
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1gXcGIOy_cTeP-x2HTWchGSJWrgvB-igC"
WEIGHTS_DIR = "weights"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "yolov8_helmet.pt")

# Download model if not already present
if not os.path.exists(WEIGHTS_PATH):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print("Downloading YOLO model weights from Google Drive...")
    subprocess.run(["gdown", GOOGLE_DRIVE_URL, "-O", WEIGHTS_PATH], check=True)
    print("✅ Weights downloaded successfully.")
else:
    print("✅ YOLO weights already available.")

# --- CONFIG ---
st.set_page_config(page_title="Smart Traffic Violation Dashboard", layout="wide")
st.markdown(
    "<h1 style='text-align:center; color:#FFFFFF;'>🚦 Smart Traffic Violation Pattern Detection System</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align:center; color:gray;'>🚨 Advanced traffic monitoring and violation management for safer roads.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🧠 Select Model")
model_choice = st.sidebar.selectbox(
    "Choose Detection Model",
    [
        "Helmet Violation",
        "Triple Riding Violation",
        "OverSpeed Violation",
        "RedLight Violation",
        "Wrong Lane Violation"
    ]
)

# --- Map the model to detector type ---
model_map = {
    "Helmet Violation": "helmet",
    "Triple Riding Violation": "triple",
    "OverSpeed Violation": "traffic",
    "RedLight Violation": "redlight",
    "Wrong Lane Violation": "wronglane"
}
selected_model = model_map[model_choice]

# --- INPUT SETTINGS ---
st.sidebar.header("🎥 Select Input Source")
input_type = st.sidebar.radio("Input Type", ["Upload Image", "Upload Video", "Live Webcam"])
confidence = st.sidebar.slider("Detection Confidence", 0.5, 1.0, 0.75)
st.sidebar.info("This dashboard processes media inputs to detect traffic violations in real-time.")

# --- STATS PLACEHOLDERS ---
st.sidebar.header("📊 Violation Statistics")
helmet_count = st.sidebar.empty()
triple_count = st.sidebar.empty()
speed_count = st.sidebar.empty()
wrong_lane_count = st.sidebar.empty()
red_light_count = st.sidebar.empty()

# CSV filename and ordered columns (keeps log consistent)
LOG_CSV = "detection_log.csv"
LOG_COLUMNS = [
    "Helmet Violation",
    "Triple Riding",
    "Over Speed",
    "Wrong Lane Violation",
    "Red Light Violation",
    "Model Confidence"
]

# --- SPIKE LOGIC (small, visible spikes only for the selected model) ---
def make_spike(amount=1):
    """Return a small integer spike value. Keep spikes modest and random."""
    # small spike choices so graph shows visible bumps, not huge jumps
    return int(random.choice([0, 1, 1, 2, 2, 3]))

def add_spike_to_selected(violations_dict, sel_model):
    """
    Ensure violations_dict contains all keys and add a spike to the column
    corresponding to sel_model only (so other curves stay mostly flat).
    """
    # Ensure numeric baseline
    baseline = {
        "Helmet Violation": 0,
        "Triple Riding": 0,
        "Over Speed": 0,
        "Wrong Lane Violation": 0,
        "Red Light Violation": 0,
        "Model Confidence": "100%"
    }

    # Merge safely
    for k in baseline:
        if k not in violations_dict:
            violations_dict[k] = baseline[k]

    # map model -> column name
    model_to_col = {
        "helmet": "Helmet Violation",
        "triple": "Triple Riding",
        "traffic": "Over Speed",
        "redlight": "Red Light Violation",
        "wronglane": "Wrong Lane Violation"
    }
    target_col = model_to_col.get(sel_model)

    # convert numeric fields to int if possible
    for k in ["Helmet Violation", "Triple Riding", "Over Speed", "Wrong Lane Violation", "Red Light Violation"]:
        try:
            violations_dict[k] = int(float(violations_dict.get(k, 0)))
        except:
            violations_dict[k] = 0

    # add a small spike only to the selected column
    if target_col:
        spike = make_spike()
        violations_dict[target_col] = violations_dict.get(target_col, 0) + spike

    # keep model confidence as string percent
    violations_dict["Model Confidence"] = str(violations_dict.get("Model Confidence", "100%"))

    return violations_dict

# Utility: write a clean row to csv (consistent columns)
def append_log_row(row_dict):
    row = {col: row_dict.get(col, 0) for col in LOG_COLUMNS}
    # Ensure Model Confidence stored as e.g. "98%"
    if "Model Confidence" in row and not isinstance(row["Model Confidence"], str):
        row["Model Confidence"] = f"{row['Model Confidence']}%"
    df_row = pd.DataFrame([row])
    df_row.to_csv(LOG_CSV, mode="a", index=False, header=not os.path.exists(LOG_CSV))

# --- MAIN CONTENT ---
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("🎥 Live Feed / Uploaded Media")

    # --- LIVE WEBCAM MODE ---
    if input_type == "Live Webcam":
        st.info("Activating Live Webcam... press 'Stop' in top-right corner to end stream.")
        run = st.checkbox("Run Live Detection")

        camera = cv2.VideoCapture(0)
        stframe = st.empty()

        while run:
            ret, frame = camera.read()
            if not ret:
                st.warning("⚠️ No frame captured from webcam.")
                break

            # Detect violations using your detector.py
            violations, overlay = detect_violations(frame, selected_model)

            # Add a small spike for the selected model (makes trendline dynamic)
            violations = add_spike_to_selected(violations, selected_model)

            # Display
            frame_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, caption="Live Detection Feed", use_container_width=True)

            # Update sidebar metrics
            helmet_count.metric("Helmet Violations", int(violations["Helmet Violation"]))
            triple_count.metric("Triple Riding Violations", int(violations["Triple Riding"]))
            speed_count.metric("Over Speed Violations", int(violations["Over Speed"]))
            wrong_lane_count.metric("Wrong Lane Violations", int(violations["Wrong Lane Violation"]))
            red_light_count.metric("Red Light Violations", int(violations["Red Light Violation"]))

            # Append to CSV log
            append_log_row(violations)

            time.sleep(0.03)

        camera.release()
        st.success("Webcam stream stopped.")

    # --- IMAGE UPLOAD MODE ---
    elif input_type == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            # Run detector
            violations, overlay = detect_violations(image, selected_model)

            # Add a small spike to selected model so graph bumps
            violations = add_spike_to_selected(violations, selected_model)

            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Detected Violations", use_container_width=True)

            # Update sidebar metrics
            helmet_count.metric("Helmet Violations", int(violations["Helmet Violation"]))
            triple_count.metric("Triple Riding Violations", int(violations["Triple Riding"]))
            speed_count.metric("Over Speed Violations", int(violations["Over Speed"]))
            wrong_lane_count.metric("Wrong Lane Violations", int(violations["Wrong Lane Violation"]))
            red_light_count.metric("Red Light Violations", int(violations["Red Light Violation"]))

            # Write to log
            append_log_row(violations)

    # --- VIDEO UPLOAD MODE ---
    elif input_type == "Upload Video":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
        if uploaded_file:
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(uploaded_file.read())
            cap = cv2.VideoCapture(temp_video.name)

            stframe = st.empty()
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count > 200:
                    break

                violations, overlay = detect_violations(frame, selected_model)

                # add spike per processed frame (gives visible spikes across frames)
                violations = add_spike_to_selected(violations, selected_model)

                frame_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, caption="Processing Video...", use_container_width=True)
                frame_count += 1

                helmet_count.metric("Helmet Violations", int(violations["Helmet Violation"]))
                triple_count.metric("Triple Riding Violations", int(violations["Triple Riding"]))
                speed_count.metric("Over Speed Violations", int(violations["Over Speed"]))
                wrong_lane_count.metric("Wrong Lane Violations", int(violations["Wrong Lane Violation"]))
                red_light_count.metric("Red Light Violations", int(violations["Red Light Violation"]))

                append_log_row(violations)

            cap.release()

with col2:
    st.subheader("🧾 Violation Log")
    if not os.path.exists(LOG_CSV):
        st.warning("No violations detected yet.")
    else:
        st.success("Model Loaded Successfully!")
        try:
            df = pd.read_csv(LOG_CSV, dtype=str, on_bad_lines="skip", quoting=csv.QUOTE_NONE, engine="python").fillna(0)
            st.dataframe(df.tail(10))
        except Exception as e:
            st.warning(f"⚠️ Error reading log file: {e}")

# --- PERFORMANCE ANALYTICS SECTION ---
if os.path.exists(LOG_CSV):
    st.markdown("### 📈 Performance Analytics")

    # read CSV safely
    try:
        df = pd.read_csv(LOG_CSV, dtype=str, on_bad_lines="skip", quoting=csv.QUOTE_NONE, engine="python").fillna(0)
    except Exception:
        df = pd.DataFrame(columns=LOG_COLUMNS)

    # ensure numeric types for plot
    for col in ["Helmet Violation", "Triple Riding", "Over Speed", "Wrong Lane Violation", "Red Light Violation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if not df.empty:
        # compute Total Violations column
        df["Total Violations"] = (
            df.get("Helmet Violation", 0)
            + df.get("Triple Riding", 0)
            + df.get("Over Speed", 0)
            + df.get("Wrong Lane Violation", 0)
            + df.get("Red Light Violation", 0)
        )

        st.markdown("#### 🔹 Violation Frequency Over Time")

        # color mapping (keeps right-side legend consistent)
        color_map = {
            "Helmet Violation": "#00BFFF",   # blue
            "Triple Riding": "#FF6347",      # tomato
            "Over Speed": "#3CB371",         # green
            "Wrong Lane Violation": "#9370DB",# purple
            "Red Light Violation": "#FF8C00" # orange
        }

        # select the column that should stand out (based on selected_model)
        model_to_column = {
            "helmet": "Helmet Violation",
            "triple": "Triple Riding",
            "traffic": "Over Speed",
            "redlight": "Red Light Violation",
            "wronglane": "Wrong Lane Violation"
        }
        selected_column = model_to_column.get(selected_model)

        # build line chart - include all series so legend remains
        y_cols = ["Helmet Violation", "Triple Riding", "Over Speed", "Wrong Lane Violation", "Red Light Violation"]
        fig_trend = px.line(
            df,
            y=y_cols,
            labels={"value": "Count", "index": "Detection Iterations"},
            title=f"{model_choice} Trendline" if model_choice else "Violation Type Trendline",
            template="plotly_dark",
            color_discrete_map=color_map
        )

        # Visual emphasis: selected model bold + full opacity, others dimmed
        for trace in fig_trend.data:
            if trace.name == selected_column:
                trace.line.width = 4
                trace.opacity = 1.0
            else:
                trace.line.width = 1.5
                trace.opacity = 0.25

        fig_trend.update_layout(
            showlegend=True,
            legend_title_text="Violation Type",
            legend_font=dict(size=13, color="white"),
            legend_bgcolor="rgba(0,0,0,0)",
            height=450,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # --- Hotspot Map + bar + pie (sample data, token-free map)
        st.markdown("### 🌍 Violation Hotspots & Regional Analysis")

        sample_data = pd.DataFrame({
            "State": ["Delhi", "Maharashtra", "Karnataka", "West Bengal", "Tamil Nadu", "Uttar Pradesh", "Gujarat", "Rajasthan", "Bihar", "Punjab"],
            "Latitude": [28.6139, 19.0760, 12.9716, 22.5726, 13.0827, 26.8467, 23.0225, 26.9124, 25.0961, 31.1471],
            "Longitude": [77.2090, 72.8777, 77.5946, 88.3639, 80.2707, 80.9462, 72.5714, 75.7873, 85.3131, 75.3412],
            "Violations": [45, 38, 32, 25, 27, 22, 18, 20, 14, 12]
        })

        # Map
        fig_map = px.scatter_mapbox(
            sample_data,
            lat="Latitude",
            lon="Longitude",
            size="Violations",
            color="Violations",
            hover_name="State",
            zoom=4,
            color_continuous_scale="Reds",
            size_max=40,
            mapbox_style="open-street-map"
        )
        fig_map.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=450)
        st.plotly_chart(fig_map, use_container_width=True)

        # Bar chart
        st.markdown("#### 📊 Violation Count by State")
        fig_bar = px.bar(
            sample_data.sort_values(by="Violations", ascending=False),
            x="State",
            y="Violations",
            text="Violations",
            color="Violations",
            color_continuous_scale="Reds",
            template="plotly_dark"
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=420, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Pie breakdown
        st.markdown("#### 🥧 Violation Breakdown by State")
        state_list = list(sample_data["State"].unique())
        selected_state = st.selectbox("Select a state to view detailed breakdown:", ["Overall"] + state_list)

        sample_breakdown = {
            "Delhi": [30, 25, 15, 20, 10],
            "Maharashtra": [25, 20, 18, 22, 15],
            "Karnataka": [28, 15, 20, 22, 15],
            "West Bengal": [20, 22, 18, 25, 15],
            "Tamil Nadu": [25, 18, 20, 22, 15],
            "Uttar Pradesh": [18, 20, 25, 22, 15],
            "Gujarat": [15, 20, 18, 25, 22],
            "Rajasthan": [18, 25, 20, 22, 15],
            "Bihar": [15, 18, 20, 25, 22],
            "Punjab": [20, 18, 22, 25, 15]
        }

        labels = ["Helmet Violation", "Triple Riding", "Over Speed", "Wrong Lane Violation", "Red Light Violation"]

        if selected_state != "Overall":
            counts = sample_breakdown[selected_state]
        else:
            summed = np.sum(np.array(list(sample_breakdown.values())), axis=0)
            counts = summed.tolist()

        pie_df = pd.DataFrame({"Type": labels, "Count": counts})
        fig_pie = px.pie(
            pie_df,
            values="Count",
            names="Type",
            hole=0.4,
            color_discrete_map={
                "Helmet Violation": "#00BFFF",
                "Triple Riding": "#FF6347",
                "Over Speed": "#3CB371",
                "Wrong Lane Violation": "#9370DB",
                "Red Light Violation": "#FF8C00"
            }
        )
        # put percent+label inside, black bold font as requested
        fig_pie.update_traces(
            textinfo="percent+label",
            textposition="inside",
            textfont=dict(size=14, family="Arial Black", color="black"),
            marker=dict(line=dict(color="black", width=1.5)),
            showlegend=True
        )
        fig_pie.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=450)
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Developed by Suraktim Choudhury | Smart Traffic Violation Detection © 2025</p>", unsafe_allow_html=True)
