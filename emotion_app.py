import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import neattext as nt

# Load model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# App Config
st.set_page_config(page_title="Text Emotion Detector", layout="centered")

st.title("🎭 Text Emotion Detection App")
st.markdown("Type or paste a sentence, and this app will detect the **emotion** behind it.")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Text Input
input_text = st.text_area("📝 Enter your text here:", height=150, placeholder="e.g., I'm feeling really great today!")

# Predict Emotion
if st.button("🔍 Detect Emotion"):
    if input_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Clean text
        clean_text = nt.TextFrame(input_text).clean_text()

        # Vectorize and predict
        vectorized_input = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized_input)[0]
        prediction_proba = model.predict_proba(vectorized_input)

        # Display prediction
        st.success(f"🎯 **Detected Emotion:** `{prediction.capitalize()}`")

        # Confidence scores
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.markdown("#### 🔢 Confidence Scores")
        st.bar_chart(proba_df.T)

        # Store in history
        st.session_state.history.append((input_text, prediction))

# History Section
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 History")
    for idx, (text, emotion) in enumerate(reversed(st.session_state.history[-5:]), start=1):
        st.write(f"{idx}. \"{text}\" — *{emotion.capitalize()}*")
