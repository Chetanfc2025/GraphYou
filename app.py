import streamlit as st
import cv2
import numpy as np
import math
import pickle
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit  # For sigmoid transformation in linear SVM

# --- Load model and scaler ---
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# --- Personality Mapping ---
personality_map = {
    0: {
        "name": "Introverted & Thoughtful",
        "description": "🧠 Prefers solitude, enjoys deep thinking, and is highly reflective. Takes time before making decisions and is more comfortable with smaller social circles."
    },
    1: {
        "name": "Outgoing & Confident",
        "description": "🎉 Sociable, enjoys engaging with others, and exudes confidence. Loves attention, thrives in social situations, and is comfortable being the center of attraction."
    },
    2: {
        "name": "Creative & Expressive",
        "description": "🎨 Imaginative, highly expressive, and values artistic expression. Enjoys exploring new ideas and thrives in creative environments."
    },
    3: {
        "name": "Analytical & Detail-Oriented",
        "description": "📊 Focused on precision, prefers logic over emotions, and excels at problem-solving. Pays close attention to detail and thrives in structured environments."
    },
    4: {
        "name": "Empathetic & Compassionate",
        "description": "❤️ Emotionally attuned, values deep connections, and is highly empathetic. Drawn toward helping others and forming strong emotional bonds."
    }
}

# --- Preprocessing ---
def preprocess_image(img):
    """Convert to grayscale and apply thresholding"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    return thresh

# --- Feature Extraction ---
def extract_all_features(image):
    """Extract all handwriting features from the uploaded image"""
    processed_img = preprocess_image(image)

    # Mock feature values (replace with actual extractions)
    features = {
        'baseline_angle': np.random.uniform(-45, 45),   # Simulated baseline angle
        'top_margin': np.random.uniform(0, 1),         # Normalized 0-1
        'letter_size': np.random.uniform(10, 50),      # 10-50 pixels
        'line_spacing': np.random.uniform(0.5, 2.0),   # Line spacing ratio
        'word_spacing': np.random.uniform(0.5, 3.0),   # Word spacing ratio
        'pen_pressure': np.random.uniform(50, 250),    # Pen pressure
        'slant_angle': np.random.uniform(-60, 60)      # Slant angle
    }
    
    # Validate features for NaN or infinite values
    for key, val in features.items():
        if np.isnan(val) or np.isinf(val):
            st.warning(f"⚠️ Invalid value for {key}. Using default 0.")
            features[key] = 0  # Safe fallback

    return features

# --- Streamlit UI ---
st.set_page_config(page_title="Graphology Analysis", layout="wide")
st.title("📝 Enhanced Graphology Analysis using Machine Learning")

# File Uploader
uploaded_file = st.file_uploader("📤 Upload a handwriting sample (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="📸 Uploaded Handwriting Sample", use_column_width=True)

    # Extract Features
    features = extract_all_features(image)

    # Prepare Features for Prediction
    feature_values = np.array([[features['baseline_angle'],
                                features['top_margin'],
                                features['letter_size'],
                                features['line_spacing'],
                                features['word_spacing'],
                                features['pen_pressure'],
                                features['slant_angle']]])

    # Apply Scaling
    scaled_features = scaler.transform(feature_values)

    # --- Debugging Panel ---
    with st.expander("🔍 Feature Debug"):
        st.write("📊 Raw Feature Values:", features)
        st.write("📈 Scaled Features:", scaled_features[0])

        # Plot feature values as bar chart
        fig, ax = plt.subplots()
        pd.DataFrame(features, index=[0]).T.plot(kind='bar', ax=ax, legend=False)
        ax.set_title("Extracted Feature Values")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # --- Prediction ---
    prediction = model.predict(scaled_features)
    
    # Handling decision scores for linear SVM since `predict_proba` is unavailable
    decision_scores = model.decision_function(scaled_features)
    probabilities = expit(decision_scores)  # Apply sigmoid to map to probability-like values

    # --- Personality Probability Chart ---
    prob_df = pd.DataFrame({
        "Personality": [personality_map[i]["name"] for i in range(5)],
        "Probability": probabilities.flatten()
    }).sort_values("Probability", ascending=False)

    st.subheader("📊 Personality Prediction Probabilities")
    st.bar_chart(prob_df.set_index("Personality"))

    # --- Predicted Personality ---
    pred_class = prediction[0]
    personality = personality_map.get(pred_class, {
        "name": "Unknown",
        "description": "❓ No matching personality found. Please check your input."
    })

    st.success(f"🎯 **Primary Personality:** {personality['name']}")
    st.write(personality['description'])

    # --- Feature Explanations ---
    st.subheader("✍️ Detailed Handwriting Characteristics")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("📏 Baseline Angle", f"{features['baseline_angle']:.1f}°")
        st.metric("🔠 Letter Size", f"{features['letter_size']:.1f} px")
        st.metric("🔡 Word Spacing", f"{features['word_spacing']:.1f} ratio")

    with col2:
        st.metric("🖊️ Slant Angle", f"{features['slant_angle']:.1f}°")
        st.metric("💡 Pen Pressure", f"{features['pen_pressure']:.1f}")
        st.metric("📄 Line Spacing", f"{features['line_spacing']:.1f} ratio")
