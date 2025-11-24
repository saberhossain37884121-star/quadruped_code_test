import cv2
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

# ================== CONFIGURATION ==================
IMG_SIZE = (128, 128)
CLASSIFICATION_CATEGORIES = ["Closed", "Open", "Semi"]

# Load your trained pipeline (you'll save this after training)
PIPELINE_PATH = "best_door_classifier_pipeline.pkl"  # You will generate this

# HOG Parameters (MUST match training)
hog = cv2.HOGDescriptor(
    _winSize=(128, 128),
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9
)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# ================== LOAD TRAINED PIPELINE ==================
print("Loading trained model pipeline...")
with open(PIPELINE_PATH, 'rb') as f:
    pipeline = pickle.load(f)

scaler = pipeline['scaler']
pca = pipeline['pca']
svm_model = pipeline['model']
print(f"Loaded SVM model: {svm_model.__class__.__name__} with {svm_model.C=}, kernel={svm_model.kernel}")

# ================== REAL-TIME PREDICTION FUNCTION ==================
def predict_door_state(frame):
    # Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    clahe_img = clahe.apply(resized)
    
    # Extract HOG
    hog_features = hog.compute(clahe_img)
    if hog_features is None:
        return None, None
    hog_vector = hog_features.flatten().reshape(1, -1)
    
    # Scale & PCA
    scaled = scaler.transform(hog_vector)
    pca_features = pca.transform(scaled)
    2
    # Predict
    pred_idx = svm_model.predict(pca_features)[0]
    confidence = svm_model.predict_proba(pca_features)[0].max()
    
    predicted_label = CLASSIFICATION_CATEGORIES[pred_idx]
    return predicted_label, confidence

# ================== REAL-TIME VIDEO CAPTURE ==================
cap = cv2.VideoCapture("http://192.168.0.146:4747/video")  # 0 = default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

print("Starting real-time door state detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # Make prediction
    label, conf = predict_door_state(frame)
    
    if label is not None:
        color = (0, 255, 0) if label == "Open" else (0, 255, 255) if label == "Semi" else (0, 0, 255)
        text = f"{label} ({conf:.2f})"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    cv2.imshow("Real-Time Door State Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()