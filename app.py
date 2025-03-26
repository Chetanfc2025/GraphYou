import streamlit as st
import cv2
import numpy as np
import math
import pickle
from PIL import Image

# Load the trained SVM model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Map class labels to personality types with detailed descriptions
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

# --- Streamlit UI ---
st.title("📝 Graphology Analysis using Machine Learning")

# File uploader
uploaded_file = st.file_uploader("Upload a handwriting sample", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Handwriting Sample", use_column_width=True)

    # Preprocess the image
    def preprocess_image(img):
        """Convert to grayscale and apply thresholding"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def estimate_baseline_angle(img):
        """Estimate the baseline angle of the handwriting"""
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

        baseline_angle = np.mean(angles) if angles else 0.0
        return baseline_angle

    def estimate_top_margin(img):
        """Estimate the top margin of the handwriting"""
        horizontal_proj = np.sum(img, axis=1)
        top_margin_index = np.argmax(horizontal_proj > np.mean(horizontal_proj))
        return top_margin_index / img.shape[0] if top_margin_index > 0 else 0.0

    def estimate_letter_size(img):
        """Simplified letter size estimation"""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        heights = [cv2.boundingRect(ctr)[3] for ctr in contours if cv2.boundingRect(ctr)[3] > 10]
        return np.mean(heights) if heights else 0

    def estimate_line_spacing(img):
        """Simplified line spacing estimation"""
        horizontal_proj = np.sum(img, axis=1)
        lines = np.where(horizontal_proj > np.mean(horizontal_proj))[0]
        spacing = np.mean(np.diff(lines)) if len(lines) > 1 else 0
        return spacing / features['letter_size'] if features['letter_size'] else 0

    def estimate_word_spacing(img):
        """Simplified word spacing estimation"""
        vertical_proj = np.sum(img, axis=0)
        words = np.where(vertical_proj > np.mean(vertical_proj))[0]
        spacing = np.mean(np.diff(words)) if len(words) > 1 else 0
        return spacing / features['letter_size'] if features['letter_size'] else 0

    def estimate_pen_pressure(img):
        """Simplified pen pressure estimation"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        non_zero = inverted[inverted > 50]
        return np.mean(non_zero) if non_zero.size > 0 else 0

    def estimate_slant_angle(img):
        """Simplified slant angle estimation"""
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:  # Filter out nearly horizontal lines
                angles.append(angle)

        return np.mean(angles) if angles else 0.0

    # Process the uploaded image
    processed_img = preprocess_image(image)

    # Extract and calculate all features
    features = {
        'baseline_angle': estimate_baseline_angle(processed_img),
        'top_margin': estimate_top_margin(processed_img),
        'letter_size': estimate_letter_size(processed_img),
        'line_spacing': estimate_line_spacing(processed_img),
        'word_spacing': estimate_word_spacing(processed_img),
        'pen_pressure': estimate_pen_pressure(image),
        'slant_angle': estimate_slant_angle(processed_img)
    }

    # Extract features for prediction
    feature_values = [
        features['baseline_angle'],
        features['top_margin'],
        features['letter_size'],
        features['line_spacing'],
        features['word_spacing'],
        features['pen_pressure'],
        features['slant_angle']
    ]

    # --- Debugging: Show feature shapes ---
    st.write(f"Feature values: {feature_values}")
    st.write(f"Model expects: {model.n_features_in_} features")

    # Check if feature values match the expected shape
    if len(feature_values) == model.n_features_in_:
        feature_values = np.array(feature_values).reshape(1, -1)
        prediction = model.predict(feature_values)

        # Map the prediction to a personality label
        predicted_personality = personality_map.get(prediction[0], {"name": "Unknown Personality", "description": "❓ No matching personality found. Please check your input."})

        # --- Display Enhanced Analysis Results ---
        st.subheader("🔎 Enhanced Handwriting Analysis Report")

        # Enhanced feature display with progress bars
        for feature, value in features.items():
            # Define dynamic range for each feature
            if feature == 'pen_pressure':
                max_value = 255  # Pen pressure can go higher
            elif feature == 'letter_size':
                max_value = 100  # Letter size typically within 100
            elif feature == 'baseline_angle' or feature == 'slant_angle':
                max_value = 90  # Angles range from -90 to +90
            elif feature in ['top_margin', 'line_spacing', 'word_spacing']:
                max_value = 1.0  # Normalized between 0 and 1
            else:
                max_value = 50  # Safe fallback for unknowns

            # Handle NaN or invalid values safely
            if np.isnan(value) or np.isinf(value):
                progress_value = 0.0
            else:
                # Safeguard progress value between 0 and 1.0
                progress_value = min(max(value / max_value, 0.0), 1.0)

            st.write(f"**{feature.replace('_', ' ').title()}**: {value:.2f}")
            st.progress(progress_value)

        # 🎭 Personality Insights Section
        st.subheader("🎭 Personality Insights")
        st.write(f"✅ **{predicted_personality['name']}**")
        st.write(f"{predicted_personality['description']}")

        st.success(f"🎯 Predicted Personality: {predicted_personality['name']}")

    else:
        st.error(f"⚠️ Feature shape mismatch! Model expects {model.n_features_in_} features, but got {len(feature_values)}.")
