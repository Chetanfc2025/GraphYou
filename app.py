import streamlit as st
import cv2
import numpy as np
import math
import pickle
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Personality mapping
personality_map = {
    0: {"name": "Introverted & Thoughtful", "description": "🧠 Prefers solitude, enjoys deep thinking..."},
    1: {"name": "Outgoing & Confident", "description": "🎉 Sociable, enjoys engaging with others..."},
    2: {"name": "Creative & Expressive", "description": "🎨 Imaginative, highly expressive..."},
    3: {"name": "Analytical & Detail-Oriented", "description": "📊 Focused on precision, prefers logic..."},
    4: {"name": "Empathetic & Compassionate", "description": "❤️ Emotionally attuned, values deep connections..."}
}

# --- Enhanced Feature Extraction ---
def preprocess_image(img):
    """Improved image preprocessing"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    return thresh

def extract_all_features(image):
    """Comprehensive feature extraction with validation"""
    processed_img = preprocess_image(image)
    
    # Feature extraction functions would go here
    # ... (your existing feature extraction code)
    
    # Mock features for demonstration - REPLACE WITH YOUR ACTUAL EXTRACTION
    features = {
        'baseline_angle': np.random.uniform(-45, 45),
        'top_margin': np.random.uniform(0, 1),
        'letter_size': np.random.uniform(10, 50),
        'line_spacing': np.random.uniform(0.5, 2.0),
        'word_spacing': np.random.uniform(0.5, 3.0),
        'pen_pressure': np.random.uniform(50, 250),
        'slant_angle': np.random.uniform(-60, 60)
    }
    
    # Validate features
    for key, val in features.items():
        if np.isnan(val) or np.isinf(val):
            st.warning(f"Invalid value for {key}: {val}")
            features[key] = 0  # Safe default
    
    return features

# --- Streamlit UI ---
st.set_page_config(page_title="Graphology Analysis", layout="wide")
st.title("📝 Enhanced Graphology Analysis")

uploaded_file = st.file_uploader("Upload handwriting sample", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Sample", use_column_width=True)
    
    # Extract features
    features = extract_all_features(image)
    
    # Prepare for prediction
    feature_values = np.array([[
        features['baseline_angle'],
        features['top_margin'],
        features['letter_size'],
        features['line_spacing'],
        features['word_spacing'],
        features['pen_pressure'],
        features['slant_angle']
    ]])
    
    # Scale features
    scaled_features = scaler.transform(feature_values)
    
    # Debug panel
    with st.expander("🔍 Feature Debug"):
        st.write("Raw Features:", features)
        st.write("Scaled Features:", scaled_features[0])
        
        fig, ax = plt.subplots()
        pd.DataFrame(features, index=[0]).T.plot(kind='bar', ax=ax)
        st.pyplot(fig)
    
    # Predict with probabilities
    prediction = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)[0]
    
    # Display results
    st.subheader("📊 Analysis Results")
    
    # Personality probabilities chart
    prob_df = pd.DataFrame({
        "Personality": [personality_map[i]["name"] for i in range(5)],
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)
    
    st.bar_chart(prob_df.set_index("Personality"))
    
    # Top prediction
    pred_class = prediction[0]
    personality = personality_map.get(pred_class, {
        "name": "Unknown", 
        "description": "Could not determine personality"
    })
    
    st.success(f"🎯 Primary Personality: {personality['name']}")
    st.write(personality['description'])
    
    # Feature explanations
    st.subheader("✍️ Handwriting Characteristics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Baseline Angle", f"{features['baseline_angle']:.1f}°")
        st.metric("Letter Size", f"{features['letter_size']:.1f} px")
        st.metric("Word Spacing", f"{features['word_spacing']:.1f} ratio")
        
    with col2:
        st.metric("Slant Angle", f"{features['slant_angle']:.1f}°")
        st.metric("Pen Pressure", f"{features['pen_pressure']:.1f}")
        st.metric("Line Spacing", f"{features['line_spacing']:.1f} ratio")