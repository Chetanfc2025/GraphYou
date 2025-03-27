from sklearn.metrics import accuracy_score
import streamlit as st
import cv2
import numpy as np
import math
import pickle
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit

# --- Page Configuration (Must Be First) ---
st.set_page_config(page_title="Graphology Analysis", layout="wide")

# --- Custom CSS for Enhanced UI ---
st.markdown(
    """
    <style>
    body, p, label, div {
        font-size: 18px !important;
        font-family: Arial, sans-serif;
    }
    h1, h2, h3 {
        font-size: 28px !important;
        color: #4a90e2;
    }
    .stMetric {
        font-size: 20px !important;
        color: #ff6347;
    }
    .stButton > button {
        background-color: #4caf50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model and Scaler ---
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- User Instructions ---
st.markdown(
    """
    ## 📚 Instructions for Uploading a Handwriting Sample
    - ✨ **Use plain white paper** without any margins.
    - 📝 **Write in English** with any content of your choice.
    - 📏 **Write more than 10 lines** for reliable results.
    - 📸 **Upload a clear image** similar to the sample provided.
    """
)

# --- Personality Mapping ---
personality_map = {
    0: {"name": "Introverted & Thoughtful", "description": "🧠 Prefers solitude, enjoys deep thinking, and is highly reflective."},
    1: {"name": "Outgoing & Confident", "description": "🎉 Sociable, enjoys engaging with others, and exudes confidence."},
    2: {"name": "Creative & Expressive", "description": "🎨 Imaginative, highly expressive, and values artistic expression."},
    3: {"name": "Analytical & Detail-Oriented", "description": "📊 Focused on precision, prefers logic over emotions."},
    4: {"name": "Empathetic & Compassionate", "description": "❤️ Emotionally attuned, values deep connections, and is highly empathetic."}
}

# --- Feature Descriptions ---
feature_descriptions = {
    "baseline_angle": {"low": "➡️ Slightly inclined writing suggests calm, stability, and control.", "high": "↘️ Highly inclined writing indicates spontaneity, impulsiveness, or creativity."},
    "letter_size": {"low": "🔍 Small letters suggest introversion, focus, and attention to detail.", "high": "🌟 Large letters indicate outgoing nature, confidence, and expressiveness."},
    "line_spacing": {"low": "👥 Closely spaced lines suggest high emotional intensity and impatience.", "high": "⏰ Widely spaced lines indicate calmness, patience, and a relaxed attitude."},
    "word_spacing": {"low": "🔒 Narrow word spacing suggests being reserved, cautious, and guarded.", "high": "🚀 Wide word spacing indicates openness, sociability, and independence."},
    "pen_pressure": {"low": "🧯 Light pressure shows sensitivity, empathy, and delicacy.", "high": "⚡ Heavy pressure indicates determination, passion, and high emotional intensity."},
    "slant_angle": {"low": "↖️ Left slant suggests introspection, emotional control, and independence.", "high": "↘️ Right slant indicates expressiveness, sociability, and emotional openness."}
}

# --- Preprocessing ---
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    return thresh

# --- Baseline Angle ---
def estimate_baseline_angle(thresh):
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            angles.append(angle)
    return np.mean(angles) if angles else 0

# --- Top Margin ---
def estimate_top_margin(thresh):
    rows = np.sum(thresh, axis=1)
    top_margin = np.argmax(rows > 0)
    return top_margin

# --- Letter Size ---
def estimate_letter_size(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(c)[3] for c in contours]
    return np.mean(heights) if heights else 0

# --- Line Spacing ---
def estimate_line_spacing(thresh):
    rows = np.sum(thresh, axis=1)
    line_indices = np.where(rows > 0)[0]
    spacings = np.diff(line_indices)
    return np.mean(spacings) if len(spacings) > 1 else 0

# --- Word Spacing ---
def estimate_word_spacing(thresh):
    cols = np.sum(thresh, axis=0)
    word_indices = np.where(cols > 0)[0]
    spacings = np.diff(word_indices)
    return np.mean(spacings) if len(spacings) > 1 else 0

# --- Pen Pressure ---
def estimate_pen_pressure(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# --- Slant Angle ---
def estimate_slant_angle(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for c in contours:
        if len(c) >= 5:
            _, _, angle = cv2.fitEllipse(c)
            angles.append(angle)
    return np.mean(angles) if angles else 0

# --- Feature Extraction ---
def extract_all_features(image):
    processed_img = preprocess_image(image)
    features = {
        'baseline_angle': estimate_baseline_angle(processed_img),
        'top_margin': estimate_top_margin(processed_img),
        'letter_size': estimate_letter_size(processed_img),
        'line_spacing': estimate_line_spacing(processed_img),
        'word_spacing': estimate_word_spacing(processed_img),
        'pen_pressure': estimate_pen_pressure(image),
        'slant_angle': estimate_slant_angle(processed_img)
    }
    for key, val in features.items():
        if np.isnan(val) or np.isinf(val):
            features[key] = 0
    return features

# --- Streamlit UI ---
st.title("📝 Decoding Emotion using Your Handwriting")

uploaded_file = st.file_uploader("📤 Upload a handwriting sample (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="📸 Uploaded Handwriting Sample", use_column_width=True)

    features = extract_all_features(image)
    feature_values = np.array([[features['baseline_angle'],
                                features['top_margin'],
                                features['letter_size'],
                                features['line_spacing'],
                                features['word_spacing'],
                                features['pen_pressure'],
                                features['slant_angle']]])
    scaled_features = scaler.transform(feature_values)
    prediction = model.predict(scaled_features)

    # --- Personality Prediction ---
    pred_class = prediction[0]
    personality = personality_map.get(pred_class, {
        "name": "Unknown",
        "description": "❓ No matching personality found. Please check your input."
    })

    st.success(f"🎯 **Primary Personality:** {personality['name']}")
    st.write(personality['description'])

    # --- Detailed Analysis ---
    st.subheader("✍️ Detailed Handwriting Characteristics with Descriptions")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("📏 Baseline Angle", f"{features['baseline_angle']:.1f}°")
        st.write(feature_descriptions["baseline_angle"]["low" if features['baseline_angle'] < 0 else "high"])

        st.metric("🔠 Letter Size", f"{features['letter_size']:.1f} px")
        st.write(feature_descriptions["letter_size"]["low" if features['letter_size'] < 20 else "high"])

        st.metric("🔡 Word Spacing", f"{features['word_spacing']:.1f} ratio")
        st.write(feature_descriptions["word_spacing"]["low" if features['word_spacing'] < 20 else "high"])

    with col2:
        st.metric("📚 Line Spacing", f"{features['line_spacing']:.1f} px")
        st.write(feature_descriptions["line_spacing"]["low" if features['line_spacing'] < 20 else "high"])

        st.metric("💪 Pen Pressure", f"{features['pen_pressure']:.1f}")
        st.write(feature_descriptions["pen_pressure"]["low" if features['pen_pressure'] < 127 else "high"])

        st.metric("🧭 Slant Angle", f"{features['slant_angle']:.1f}°")
        st.write(feature_descriptions["slant_angle"]["low" if features['slant_angle'] < 45 else "high"])
