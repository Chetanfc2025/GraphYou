import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# --- Sample Data Generation ---
# Replace this with your actual feature values and labels if you have them
np.random.seed(42)
X = np.random.rand(100, 7)  # 100 samples, 7 features each
y = np.random.choice([0, 1], size=100)  # 0 or 1 as target labels (binary classification)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Save the trained SVM model to 'model.pkl'
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ New SVM model trained successfully with 7 features and saved as 'model.pkl'!")
