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
    "pen_pressure": {"low": "🪶 Light pressure shows sensitivity, empathy, and delicacy.", "high": "⚡ Heavy pressure indicates determination, passion, and high emotional intensity."},
    "slant_angle": {"low": "↖️ Left slant suggests introspection, emotional control, and independence.", "high": "↘️ Right slant indicates expressiveness, sociability, and emotional openness."}
}

# --- Preprocessing ---
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    return thresh

# --- Feature Extraction (Same as Original) ---

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
    decision_scores = model.decision_function(scaled_features)
    probabilities = expit(decision_scores)

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
        st.write(feature_descriptions["word_spacing"]["low" if features['word_spacing'] < 1 else "high"])

    with col2:
        st.metric("🖊️ Slant Angle", f"{features['slant_angle']:.1f}°")
        st.write(feature_descriptions["slant_angle"]["low" if features['slant_angle'] < 0 else "high"])

        st.metric("💡 Pen Pressure", f"{features['pen_pressure']:.1f}")
        st.write(feature_descriptions["pen_pressure"]["low" if features['pen_pressure'] < 150 else "high"])

        st.metric("📄 Line Spacing", f"{features['line_spacing']:.1f} ratio")
        st.write(feature_descriptions["line_spacing"]["low" if features['line_spacing'] < 1 else "high"])

    # --- Personality Probability Chart ---
    prob_df = pd.DataFrame({
        "Personality": [personality_map[i]["name"] for i in range(5)],
        "Probability": probabilities.flatten()
    }).sort_values("Probability", ascending=False)

    st.subheader("📊 Personality Prediction Probabilities")
    st.bar_chart(prob_df.set_index("Personality"))
