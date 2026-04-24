from streamlit_autorefresh import st_autorefresh
import csv
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Posture Dashboard", layout="wide")

st.title("AI Posture Correction Dashboard")
st_autorefresh(interval=10000, key="dashboard_refresh")

log_path = Path("logs/posture_log.csv")

if not log_path.exists():
    st.warning("No posture log found yet. Run posture_monitor.py first.")
    st.stop()

df = pd.read_csv(log_path)
df= df.tail(50)

if df.empty:
    st.warning("The log file is empty. Run posture_monitor.py for a few seconds.")
    st.stop()

df["is_bad_posture"] = df["is_bad_posture"].astype(int)
df["neck_angle"] = df["neck_angle"].astype(float)
df["torso_angle"] = df["torso_angle"].astype(float)

bad_count = int(df["is_bad_posture"].sum())
total_records = len(df)
bad_percentage = (bad_count / total_records) * 100

if bad_percentage < 20:
    status = "Excellent 🟢"
elif bad_percentage < 50:
    status = "Average 🟡"
else:
    status = "Bad 🔴"

st.subheader(f"Posture Status: {status}")

score = max(0, 100 - bad_percentage)

st.metric("Posture Score", f"{score:.0f}/100") 

if score >= 80:
    feedback = "Excellent posture! Keep it up 💪"
elif score >= 50:
    feedback = "Not bad, but you can improve 👍"
else:
    feedback = "Warning! Fix your posture ⚠️"

st.info(feedback)


col1, col2, col3 = st.columns(3)

col1.metric("Bad Posture Count", bad_count)
col2.metric("Total Records", total_records)
col3.metric("Bad Posture %", f"{bad_percentage:.1f}%")

col4, col5 = st.columns(2)

col4.metric("Average Neck Angle", f"{df['neck_angle'].mean():.1f}°")
col5.metric("Average Torso Angle", f"{df['torso_angle'].mean():.1f}°")

st.subheader("Posture Angles Over Time")
st.line_chart(df[["neck_angle", "torso_angle"]])

st.subheader("Bad Posture Records")
st.bar_chart(df["is_bad_posture"])

st.subheader("Recent Records")
st.dataframe(df.tail(20), use_container_width=True)