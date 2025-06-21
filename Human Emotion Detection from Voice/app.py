# app.py
import streamlit as st
import sounddevice as sd
import wavio
import joblib
import numpy as np
from feature_extraction import extract_features

st.set_page_config(page_title="Voice Emotion Detector")
st.title("🎙️ Human Emotion Detection from Voice")
st.write("Record your voice and detect the underlying emotion.")

model = joblib.load("emotion_model.pkl")

duration = 4  # in seconds
fs = 44100

if st.button("🎤 Record Voice"):
    st.info("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write("temp.wav", recording, fs, sampwidth=2)
    st.audio("temp.wav", format='audio/wav')

    features = extract_features("temp.wav").reshape(1, -1)
    prediction = model.predict(features)[0]
    st.success(f"🔊 Detected Emotion: **{prediction.upper()}**")

    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append(prediction)

    st.write("📊 Emotion Trend")
    emotion_list = list(set(st.session_state["history"]))
    freq = [st.session_state["history"].count(e) for e in emotion_list]
    st.bar_chart(data=dict(zip(emotion_list, freq)))

    with open("session_log.txt", "a") as f:
        f.write(prediction + "\n")
