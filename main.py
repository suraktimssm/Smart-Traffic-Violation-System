import pandas as pd
data = pd.read_csv(r"C:\Users\suraktim choudhury\Desktop\Smart Traffic Violation Pattern Detection\Indian_Traffic_Violations.csv")
print(data.head())   # Shows first 5 rows
print(data.info())   # Shows column details

import os
import streamlit as st

st.set_page_config(page_title="Smart Traffic Violation Detection", layout="wide")

st.title("🚦 Smart Traffic Violation Pattern Detection")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Analysis", "Model Training", "Visualization"])

if page == "Data Analysis":
    st.write("Run `data_analysis.py` to see dataset insights.")
    st.code("python data_analysis.py")

elif page == "Model Training":
    st.write("Run `model_training.py` to train the ML model.")
    st.code("python model_training.py")

elif page == "Visualization":
    st.write("Run `visualization.py` to explore the dashboard.")
    st.code("streamlit run visualization.py")
