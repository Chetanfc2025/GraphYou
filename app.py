import streamlit as st
import cv2
import numpy as np
import math
import pickle
from PIL import Image

# Constants
ANCHOR_POINT = 6000
MIDZONE_THRESHOLD = 15000

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

# --- Image Processing Functions ---
def bilateralFilter(image, d):
    image = cv2.bilateralFilter(image, d, 50, 50)
    return image

def medianFilter(image, d):
    image = cv2.medianBlur(image, d)
    return image

def threshold(image, t):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, t, 255, cv2.THRESH_BINARY_INV)
    return image

def dilate(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return image

def erode(image, kernalSize):
    kernel = np.ones(kernalSize, np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image

def horizontalProjection(img):
    (h, w) = img.shape[:2]
    sumRows = []
    for j in range(h):
        row = img[j:j+1, 0:w]
        sumRows.append(np.sum(row))
    return sumRows

def verticalProjection(img):
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j+1]
        sumCols.append(np.sum(col))
    return sumCols

def straighten(image):
    angle = 0.0
    angle_sum = 0.0
    countour_count = 0

    filtered = bilateralFilter(image, 3)
    thresh = threshold(filtered, 120)
    dilated = dilate(thresh, (5, 100))
    
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)

        if h > w or h < 20:
            continue

        roi = image[y:y+h, x:x+w]
        
        if w < image.shape[1]/2:
            roi = 255
            image[y:y+h, x:x+w] = roi
            continue

        rect = cv2.minAreaRect(ctr)
        angle = rect[2]
        
        if angle < -45.0:
            angle += 90.0

        rot = cv2.getRotationMatrix2D(((x+w)/2, (y+h)/2), angle, 1)
        extract = cv2.warpAffine(roi, rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        image[y:y+h, x:x+w] = extract
        
        angle_sum += angle
        countour_count += 1

    mean_angle = angle_sum / countour_count if countour_count > 0 else 0.0
    return image, mean_angle

def extractLines(img):
    filtered = bilateralFilter(img, 5)
    thresh = threshold(filtered, 160)
    hpList = horizontalProjection(thresh)

    topMarginCount = 0
    for sum in hpList:
        if (sum <= 255):
            topMarginCount += 1
        else:
            break

    lineTop = 0
    lineBottom = 0
    spaceTop = 0
    spaceBottom = 0
    indexCount = 0
    setLineTop = True
    setSpaceTop = True
    includeNextSpace = True
    space_zero = []
    lines = []

    for i, sum in enumerate(hpList):
        if (sum == 0):
            if (setSpaceTop):
                spaceTop = indexCount
                setSpaceTop = False
            indexCount += 1
            spaceBottom = indexCount
            if (i < len(hpList)-1):
                if (hpList[i+1] == 0):
                    continue
            if (includeNextSpace):
                space_zero.append(spaceBottom-spaceTop)
            else:
                if (len(space_zero) == 0):
                    previous = 0
                else:
                    previous = space_zero.pop()
                space_zero.append(previous + spaceBottom-lineTop)
            setSpaceTop = True

        if (sum > 0):
            if (setLineTop):
                lineTop = indexCount
                setLineTop = False
            indexCount += 1
            lineBottom = indexCount
            if (i < len(hpList)-1):
                if (hpList[i+1] > 0):
                    continue
            if (lineBottom-lineTop < 20):
                includeNextSpace = False
                setLineTop = True
                continue
            includeNextSpace = True
            lines.append([lineTop, lineBottom])
            setLineTop = True

    space_nonzero_row_count = 0
    midzone_row_count = 0
    lines_having_midzone_count = 0
    flag = False
    
    for i, line in enumerate(lines):
        segment = hpList[int(line[0]):int(line[1])]
        for j, sum in enumerate(segment):
            if (sum < MIDZONE_THRESHOLD):
                space_nonzero_row_count += 1
            else:
                midzone_row_count += 1
                flag = True

        if (flag):
            lines_having_midzone_count += 1
            flag = False

    if (lines_having_midzone_count == 0):
        lines_having_midzone_count = 1

    total_space_row_count = space_nonzero_row_count + np.sum(space_zero[1:-1])
    average_line_spacing = float(total_space_row_count) / lines_having_midzone_count
    average_letter_size = float(midzone_row_count) / lines_having_midzone_count
    
    if (average_letter_size == 0):
        average_letter_size = 1
        
    relative_line_spacing = average_line_spacing / average_letter_size
    relative_top_margin = float(topMarginCount) / average_letter_size

    return lines, average_letter_size, relative_line_spacing, relative_top_margin

def extractWords(image, lines, letter_size):
    filtered = bilateralFilter(image, 5)
    thresh = threshold(filtered, 180)
    width = thresh.shape[1]
    space_zero = []
    words = []

    for i, line in enumerate(lines):
        extract = thresh[int(line[0]):int(line[1]), 0:width]
        vp = verticalProjection(extract)

        wordStart = 0
        wordEnd = 0
        spaceStart = 0
        spaceEnd = 0
        indexCount = 0
        setWordStart = True
        setSpaceStart = True
        includeNextSpace = True
        spaces = []

        for j, sum in enumerate(vp):
            if (sum == 0):
                if (setSpaceStart):
                    spaceStart = indexCount
                    setSpaceStart = False
                indexCount += 1
                spaceEnd = indexCount
                if (j < len(vp)-1):
                    if (vp[j+1] == 0):
                        continue

                if ((spaceEnd-spaceStart) > int(letter_size/2)):
                    spaces.append(spaceEnd-spaceStart)

                setSpaceStart = True

            if (sum > 0):
                if (setWordStart):
                    wordStart = indexCount
                    setWordStart = False
                indexCount += 1
                wordEnd = indexCount
                if (j < len(vp)-1):
                    if (vp[j+1] > 0):
                        continue

                count = 0
                for k in range(int(line[1])-int(line[0])):
                    row = thresh[line[0]+k:line[0]+k+1, wordStart:wordEnd]
                    if (np.sum(row)):
                        count += 1
                if (count > int(letter_size/2)):
                    words.append([line[0], line[1], wordStart, wordEnd])

                setWordStart = True

        space_zero.extend(spaces[1:-1])

    space_columns = np.sum(space_zero)
    space_count = len(space_zero)
    if (space_count == 0):
        space_count = 1
    average_word_spacing = float(space_columns) / space_count
    if (letter_size == 0):
        letter_size = 1
    relative_word_spacing = average_word_spacing / letter_size

    return words, relative_word_spacing

def extractSlant(image, words):
    theta = [-0.785398, -0.523599, -0.261799, -0.0872665,
             0.01, 0.0872665, 0.261799, 0.523599, 0.785398]
    
    s_function = [0.0] * 9
    count_ = [0]*9

    filtered = bilateralFilter(image, 5)
    thresh = threshold(filtered, 180)

    for i, angle in enumerate(theta):
        s_temp = 0.0
        count = 0

        for j, word in enumerate(words):
            original = thresh[int(word[0]):int(word[1]), int(word[2]):int(word[3])]
            height = int(word[1])-int(word[0])
            width = int(word[3]) - int(word[2])
            shift = (math.tan(angle) * height) / 2
            pad_length = abs(int(shift))

            blank_image = np.zeros((height, width+pad_length*2, 3), np.uint8)
            new_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
            new_image[:, pad_length:width+pad_length] = original

            (height, width) = new_image.shape[:2]
            x1 = width/2
            y1 = 0
            x2 = width/4
            y2 = height
            x3 = 3*width/4
            y3 = height

            pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3]])
            pts2 = np.float32([[x1+shift, y1], [x2-shift, y2], [x3-shift, y3]])
            M = cv2.getAffineTransform(pts1, pts2)
            deslanted = cv2.warpAffine(new_image, M, (width, height))

            vp = verticalProjection(deslanted)

            for k, sum in enumerate(vp):
                if (sum == 0):
                    continue

                num_fgpixel = sum / 255

                if (num_fgpixel < int(height/3)):
                    continue

                column = deslanted[0:height, k:k+1]
                column = column.flatten()

                for l, pixel in enumerate(column):
                    if (pixel == 0):
                        continue
                    break
                for m, pixel in enumerate(column[::-1]):
                    if (pixel == 0):
                        continue
                    break

                delta_y = height - (l+m)
                h_sq = (float(num_fgpixel)/delta_y)**2
                h_wted = (h_sq * num_fgpixel) / height
                s_temp += h_wted
                count += 1

        s_function[i] = s_temp
        count_[i] = count

    max_value = 0.0
    max_index = 4
    for index, value in enumerate(s_function):
        if (value > max_value):
            max_value = value
            max_index = index

    if (max_index == 0):
        angle = 45
    elif (max_index == 1):
        angle = 30
    elif (max_index == 2):
        angle = 15
    elif (max_index == 3):
        angle = 5
    elif (max_index == 5):
        angle = -5
    elif (max_index == 6):
        angle = -15
    elif (max_index == 7):
        angle = -30
    elif (max_index == 8):
        angle = -45
    elif (max_index == 4):
        p = s_function[4] / s_function[3]
        q = s_function[4] / s_function[5]
        if ((p <= 1.2 and q <= 1.2) or (p > 1.4 and q > 1.4)):
            angle = 0
        elif ((p <= 1.2 and q-p > 0.4) or (q <= 1.2 and p-q > 0.4)):
            angle = 0
        else:
            angle = 180

    return angle

def estimate_pen_pressure(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    non_zero = inverted[inverted > 50]
    return np.mean(non_zero) if non_zero.size > 0 else 0

# --- Streamlit UI ---
st.set_page_config(
    page_title="Graphology Analysis",
    page_icon="✍️",
    layout="wide"
)

st.title("📝 Graphology Analysis using Machine Learning")

with st.sidebar:
    st.markdown("""
    ## 📝 How to Use
    1. Upload a clear image of handwriting
    2. Ensure the text is horizontal
    3. For best results, use 3-4 lines of text
    4. Avoid shadows or glare on the paper
    """)

uploaded_file = st.file_uploader("Upload a handwriting sample", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Handwriting Sample", use_column_width=True)
    
    # Process the image
    straightened_img, baseline_angle = straighten(image.copy())
    lines, letter_size, line_spacing, top_margin = extractLines(straightened_img.copy())
    words, word_spacing = extractWords(straightened_img.copy(), lines, letter_size)
    slant_angle = extractSlant(straightened_img.copy(), words)
    pen_pressure = estimate_pen_pressure(image)
    
    # Prepare features for prediction
    features = {
        'baseline_angle': baseline_angle,
        'top_margin': top_margin,
        'letter_size': letter_size,
        'line_spacing': line_spacing,
        'word_spacing': word_spacing,
        'pen_pressure': pen_pressure,
        'slant_angle': slant_angle
    }
    
    feature_values = [
        features['baseline_angle'],
        features['top_margin'],
        features['letter_size'],
        features['line_spacing'],
        features['word_spacing'],
        features['pen_pressure'],
        features['slant_angle']
    ]

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
                max_value = 255
            elif feature == 'letter_size':
                max_value = 100
            elif feature == 'baseline_angle' or feature == 'slant_angle':
                max_value = 90
            elif feature in ['top_margin', 'line_spacing', 'word_spacing']:
                max_value = 1.0
            else:
                max_value = 50

            # Handle NaN or invalid values safely
            if np.isnan(value) or np.isinf(value):
                progress_value = 0.0
            else:
                progress_value = min(max(value / max_value, 0.0), 1.0)

            st.write(f"**{feature.replace('_', ' ').title()}**: {value:.2f}")
            st.progress(progress_value)

        # Personality Insights Section
        st.subheader("🎭 Personality Insights")
        st.write(f"✅ **{predicted_personality['name']}**")
        st.write(f"{predicted_personality['description']}")

        st.success(f"🎯 Predicted Personality: {predicted_personality['name']}")

    else:
        st.error(f"⚠️ Feature shape mismatch! Model expects {model.n_features_in_} features, but got {len(feature_values)}.")
        