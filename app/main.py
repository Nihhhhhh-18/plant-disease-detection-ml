import streamlit as st
import os
import json
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf

# -------------------------------------------
# Config - adjust paths if needed
# -------------------------------------------
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(WORKING_DIR, "ml_artifacts")

FEATURE_EXTRACTOR_PATH = os.path.join(ARTIFACT_DIR, "mobilenet_feature_extractor.h5")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
LOGREG_PATH = os.path.join(ARTIFACT_DIR, "logreg_model.pkl")
RF_PATH = os.path.join(ARTIFACT_DIR, "rf_model.pkl")
SVM_PATH = os.path.join(ARTIFACT_DIR, "svm_model.pkl")   # LinearSVC pickle
PCA_PATH = os.path.join(ARTIFACT_DIR, "pca.pkl")         # PCA used for SVM
CLASS_INDICES_PATH = os.path.join(ARTIFACT_DIR, "class_indices.json")

# -------------------------------------------
# Helpers: safe file checks and loaders
# -------------------------------------------
def file_exists(path):
    return path is not None and os.path.exists(path)

@st.cache_resource
def load_keras_model(path):
    if not file_exists(path):
        raise FileNotFoundError(f"Feature extractor not found at: {path}")
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_pickle(path):
    if not file_exists(path):
        raise FileNotFoundError(f"Pickle file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_json(path):
    if not file_exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

# Try to load everything and present a clear error if something is missing
try:
    feature_extractor = load_keras_model(FEATURE_EXTRACTOR_PATH)
    scaler = load_pickle(SCALER_PATH)
    logreg = load_pickle(LOGREG_PATH)
    rf = load_pickle(RF_PATH)
    svm = None
    pca = None
    if file_exists(SVM_PATH):
        svm = load_pickle(SVM_PATH)
    if file_exists(PCA_PATH):
        pca = load_pickle(PCA_PATH)
    index_to_class = load_json(CLASS_INDICES_PATH)  # mapping: index (str) -> class name
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# -------------------------------------------
# Disease solutions mapping (your mapping)
# -------------------------------------------
disease_solutions = {
    "Apple___Apple_scab": "Remove infected leaves, use sulfur/copper fungicides, improve air circulation.",
    "Apple___Black_rot": "Prune cankers, remove mummified fruits, apply captan or thiophanate-methyl fungicide.",
    "Apple___Cedar_apple_rust": "Remove nearby junipers, apply myclobutanil or other rust fungicides.",
    "Apple___healthy": "No disease detected. Maintain proper watering and fertilization.",
    "Blueberry___healthy": "No disease detected. Maintain soil acidity and proper irrigation.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur-based fungicides and ensure proper spacing.",
    "Cherry_(including_sour)___healthy": "No disease detected. Maintain regular pruning.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant hybrids, rotate crops, apply strobilurin fungicides.",
    "Corn_(maize)___Common_rust": "Plant resistant varieties, apply fungicides if severe.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use disease-resistant hybrids and rotate crops.",
    "Corn_(maize)___healthy": "No disease detected. Continue normal field management.",
    "Grape___Black_rot": "Prune infected vines, remove mummified berries, apply mancozeb or myclobutanil.",
    "Grape___Esca_(Black_Measles)": "Remove infected wood, avoid vine stress, apply fungicides early.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Improve air flow, avoid overhead irrigation, apply mancozeb.",
    "Grape___healthy": "No disease detected. Maintain good canopy management.",
    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees, control psyllids, apply nutrition therapy.",
    "Peach___Bacterial_spot": "Use copper sprays, avoid overhead irrigation, plant resistant cultivars.",
    "Peach___healthy": "No disease detected. Continue good orchard practices.",
    "Pepper,_bell___Bacterial_spot": "Use copper-based bactericides, avoid water splash, plant resistant varieties.",
    "Pepper,_bell___healthy": "No disease detected. Maintain soil nutrients.",
    "Potato___Early_blight": "Use chlorothalonil or mancozeb fungicides, rotate crops, improve aeration.",
    "Potato___Late_blight": "Apply metalaxyl or mancozeb, destroy infected plants, avoid wet foliage.",
    "Potato___healthy": "No disease detected. Maintain proper watering.",
    "Raspberry___healthy": "No disease detected.",
    "Soybean___healthy": "No disease detected.",
    "Squash___Powdery_mildew": "Use neem oil or sulfur sprays, improve airflow, water at soil level.",
    "Strawberry___Leaf_scorch": "Remove infected leaves, improve spacing, apply balanced fertilizer.",
    "Strawberry___healthy": "No disease detected.",
    "Tomato___Bacterial_spot": "Use copper sprays, avoid leaf wetness, rotate crops.",
    "Tomato___Early_blight": "Apply chlorothalonil, remove infected stems, avoid overhead watering.",
    "Tomato___Late_blight": "Use mancozeb/metalaxyl, remove affected plants, improve airflow.",
    "Tomato___Leaf_Mold": "Reduce humidity, improve ventilation, use chlorothalonil.",
    "Tomato___Septoria_leaf_spot": "Remove infected leaves, apply copper or chlorothalonil sprays.",
    "Tomato___Spider_mites_Two_spotted_spider_mite": "Use miticides or insecticidal soap, keep plants hydrated.",
    "Tomato___Target_Spot": "Apply preventive fungicides, avoid excessive nitrogen.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies, remove infected plants.",
    "Tomato___Tomato_mosaic_virus": "Disinfect tools, avoid tobacco contamination.",
    "Tomato___healthy": "No disease detected."
}

# -------------------------------------------
# Image preprocessing & feature extraction
# -------------------------------------------
IMG_SIZE = (224, 224)

def preprocess_image(file_or_pil):
    """Return a (1, H, W, 3) numpy array scaled to [0,1]."""
    if isinstance(file_or_pil, str):
        img = Image.open(file_or_pil).convert('RGB')
    else:
        img = Image.open(file_or_pil).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def extract_features(img_arr):
    """Return scaled feature vector (1, feat_dim)"""
    feats = feature_extractor.predict(img_arr)
    feats_scaled = scaler.transform(feats)
    return feats_scaled

def softmax(x):
    x = np.array(x, dtype=np.float64)
    if x.ndim == 1:
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

# -------------------------------------------
# Streamlit UI
# -------------------------------------------
st.set_page_config(page_title="Plant Disease Predictor (ML)", layout="centered")
st.title("Plant Disease Predictor")

st.write("Upload a plant leaf image. Choose ML model for prediction.")

uploaded = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
model_choice = st.selectbox("Select model", ["Random Forest", "Logistic Regression", "SVM (with PCA)"])

# Inform if SVM/PCA not available
if model_choice == "SVM (with PCA)":
    if svm is None or pca is None:
        st.warning("SVM or PCA artifact not found. Please place 'svm_model.pkl' and 'pca.pkl' inside ml_artifacts/ to enable SVM.")
        # disable classify button by not showing it
        if uploaded is None:
            st.info("Upload an image to test other models.")
        else:
            st.image(Image.open(uploaded).resize((300,300)))
        st.stop()

if uploaded is not None:
    st.image(Image.open(uploaded).resize((300,300)))
    if st.button("Classify"):
        with st.spinner("Extracting features and predicting..."):
            # 1) preprocess and extract scaled features
            img_arr = preprocess_image(uploaded)
            feats_scaled = extract_features(img_arr)  # shape (1, feat_dim)

            # 2) model-specific predictions -> we will compute:
            #    - probs: 1D array of length n_classes aligned with model.classes_
            #    - classes_order: array-like of class labels (these are numeric labels used during training)
            probs = None
            classes_order = None
            pred_label = None  # numeric label (as used during training, e.g., 0..N-1)

            if model_choice == "Random Forest":
                # RF supports predict_proba; columns correspond to rf.classes_
                probs = rf.predict_proba(feats_scaled)[0]
                classes_order = rf.classes_
                pred_label = int(rf.predict(feats_scaled)[0])

            elif model_choice == "Logistic Regression":
                probs = logreg.predict_proba(feats_scaled)[0]
                classes_order = logreg.classes_
                pred_label = int(logreg.predict(feats_scaled)[0])

            else:  # SVM (with PCA) — LinearSVC (no predict_proba)
                # Reduce features using PCA, then use svm.decision_function
                feats_reduced = pca.transform(feats_scaled)
                scores = svm.decision_function(feats_reduced)
                # decision_function returns shape (1, n_classes) or (1,) for binary
                if scores.ndim == 1:
                    scores = scores
                else:
                    scores = scores[0]
                probs = softmax(scores)
                classes_order = svm.classes_
                # predicted label in original training label space:
                pred_label = int(classes_order[np.argmax(probs)])

            # 3) Map predicted numeric label -> human-readable classname using index_to_class
            # index_to_class maps strings of numeric indices (as saved from generator) to names
            # Convert pred_label to string key; if missing, show clear error
            key = str(pred_label)
            if key not in index_to_class:
                st.error(f"Predicted label '{pred_label}' not found in class mapping. Check that the model and class_indices.json were created from the same dataset ordering.")
                st.stop()
            pred_class_name = index_to_class[key]

            # 4) Display results
            st.success(f"Prediction: {pred_class_name}")

            # Top-3 predictions: need to map probability indices -> numeric labels -> class names
            top3_idx = np.argsort(probs)[::-1][:3]
            st.subheader("Top predictions")
            for pos in top3_idx:
                numeric_label = int(classes_order[pos])
                class_name = index_to_class.get(str(numeric_label), f"label_{numeric_label}")
                st.write(f"{class_name} — {probs[pos]*100:.2f}%")

            # 5) Show recommended solution if available
            solution = disease_solutions.get(pred_class_name, "No treatment information available.")
            st.subheader("Recommended Treatment / Solution")
            st.info(solution)

st.markdown("---")
st.caption("Models supported: Random Forest, Logistic Regression, LinearSVC (with PCA). Feature extractor: MobileNetV2.")
