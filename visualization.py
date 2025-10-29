import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Traffic Violation Analysis", layout="wide")

file_path = r"C:\Users\suraktim choudhury\Desktop\Smart Traffic Violation Pattern Detection\Indian_Traffic_Violations.csv"
df = pd.read_csv(file_path)

st.title("🚦 Smart Traffic Violation Pattern Detection")

st.sidebar.header("Filter Options")
state = st.sidebar.selectbox("Select State", df['State'].unique())
filtered = df[df['State'] == state]

st.subheader(f"Violation Overview - {state}")
fig = px.histogram(filtered, x='Violation Type', color='Weather', title="Violations by Type and Weather")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.box(filtered, x='Violation Type', y='Fine Amount', title="Fine Distribution by Violation Type")
st.plotly_chart(fig2, use_container_width=True)
