import streamlit as st
st.set_page_config(page_title="Apple Classification", layout="wide")

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import time
import os
import gdown
import torch  # For CNN/MLP models
import joblib  # For XGB models

# --------------------------------------
# Global Variables and Class Names
# --------------------------------------
class_names = ['barnae_apple', 'crimson_snow_apples', 'fuji_apple', 'gala_apple',
               'green_apple', 'pink_lady_apple', 'red_delicious_apple', 'rotten_Apple']

# --------------------------------------
# Preprocessing Functions
# --------------------------------------
def apply_clahe(image):
    # Convert from RGB to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Create CLAHE object (parameters can be tuned)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # Merge channels and convert back to RGB
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def apply_noise_reduction(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def apply_white_balance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    # Adjust the a and b channels by shifting towards neutral (128)
    a = a - (avg_a - 128)
    b = b - (avg_b - 128)
    # Clip the values to [0, 255] and convert to uint8
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    balanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2RGB)

def get_preprocessing_func(model_key):
    if "CLAHE" in model_key:
        return apply_clahe
    elif "Noise Reduction" in model_key:
        return apply_noise_reduction
    elif "White Balancing" in model_key:
        return apply_white_balance
    else:
        return lambda x: x  # No preprocessing

# --------------------------------------
# Model File IDs and File Names
# --------------------------------------
model_file_ids = {
    "CNN1 (X.P)": "1-4C217rhypyZjXoBcFLHITvYHWY_xHoE",
    "CNN2 (CLAHE)": "1-2fEm_GDoQBKFZQEik-QKYJFBYOV1G9Y",
    "CNN3 (Noise Reduction)": "1-2mliuqVwXO1TgSB74EeaXsSJnxPr0Hp",
    "CNN4 (White Balancing)": "1-2C7HCBke5lzVRcGfLFf1KxqHBYNorQb",
    "MLP1 (X.P)": "1--w_x_Pj-2-wA4-Oeo8d5kPb7k-c5aO4",
    "MLP2 (CLAHE)": "1-QFHWsiCb4PBTFZM-j4tvQdiVLsjJODz",
    "MLP3 (Noise Reduction)": "1-Y4kpV_9wsaI_dc2nSCHCi6QlYzqNqcc",
    "MLP4 (White Balancing)": "1-kcHrwX36Q5lQQQcjtfAYYykJME9p2yz",
    "XGB1 (X.P)": "1k8To7HkGtJNW1S-MTfSxWAtLxDNMvRrm",
    "XGB2 (CLAHE)": "1l_66uTpTRhjv99he-fy8BTH8JOsE_LVW",
    "XGB3 (Noise Reduction)": "1_8QQTCCTydQgMY4bkokw4zZXMCLRpa2y",
    "XGB4 (White Balancing)": "1w6lMXabD-FWKz25Jbtq5lBKkWHC6rck5"
}

model_file_names = {
    "CNN1 (X.P)": "CNN1_XP_20.pth",
    "CNN2 (CLAHE)": "CNN2_CLAHE_13.pth",
    "CNN3 (Noise Reduction)": "CNN3_NR_10.pth",
    "CNN4 (White Balancing)": "CNN4_WB_13.pth",
    "MLP1 (X.P)": "MLP1_XP_15.pth",
    "MLP2 (CLAHE)": "MLP2_HE_CLA13.pth",
    "MLP3 (Noise Reduction)": "MLP3_NR_10.pth",
    "MLP4 (White Balancing)": "MLP4_WB_17.pth",
    "XGB1 (X.P)": "xgb1_model_xp.joblib",
    "XGB2 (CLAHE)": "xgb2_model_clahe.joblib",
    "XGB3 (Noise Reduction)": "xgb3_model_nr.joblib",
    "XGB4 (White Balancing)": "xgb4_model_wb.joblib"
}

# --------------------------------------
# Model Downloading and Loading
# --------------------------------------
def download_model(model_key):
    file_id = model_file_ids[model_key]
    file_name = model_file_names[model_key]
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(file_name):
        st.write(f"Downloading model **{file_name}** ...")
        gdown.download(url, output=file_name, quiet=False)
    else:
        st.write(f"Model **{file_name}** already downloaded.")
    return file_name

@st.cache_resource
def load_model(model_key):
    file_path = download_model(model_key)
    st.write(f"Loading model from **{file_path}** ...")
    time.sleep(2)
    # For demonstration, return a dictionary with model info.
    # Replace the code below with your actual model-loading logic.
    return {"model_key": model_key, "file_path": file_path}

# --------------------------------------
# Actual Inference Function
# --------------------------------------
def actual_inference(model, input_data):
    """
    Perform actual inference using the loaded model.
    For CNN/MLP models, we assume a PyTorch model.
    For XGB models, we assume a joblib-loaded model.
    
    Replace the placeholder code with your actual model architecture and inference logic.
    """
    model_key = model["model_key"]
    file_path = model["file_path"]
    
    if model_key.startswith("CNN") or model_key.startswith("MLP"):
        # Example placeholder for PyTorch models:
        # Define and instantiate your model architecture here.
        # For instance:
        # from my_model_definitions import MyModel
        # net = MyModel()
        # net.load_state_dict(torch.load(file_path, map_location="cpu"))
        # net.eval()
        # with torch.no_grad():
        #     output = net(input_data)
        #     probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        #
        # For now, we generate random probabilities as a placeholder.
        prob = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
        return dict(zip(class_names, prob))
    
    elif model_key.startswith("XGB"):
        # For XGB models, load using joblib and use predict_proba
        xgb_model = joblib.load(file_path)
        probabilities = xgb_model.predict_proba(input_data)[0]
        return dict(zip(class_names, probabilities))
    
    else:
        # Fallback: return random probabilities
        prob = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
        return dict(zip(class_names, prob))

# --------------------------------------
# Sidebar Navigation
# --------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"

def set_page(page):
    st.session_state.page = page

st.sidebar.title("Navigation")
if st.sidebar.button("Home"):
    set_page("Home")
if st.sidebar.button("Detection"):
    set_page("Detection")

# --------------------------------------
# Main Content
# --------------------------------------
if st.session_state.page == "Home":
    st.title("Apple Types and Qualities Classification")
    st.markdown("""
    **Welcome!**
    
    This application uses advanced machine learning algorithms to classify different types and qualities of apples.
    
    **Models Available:**
    - **XGB**: eXtreme Gradient Boosting
    - **MLP**: Multi-Layer Perceptron
    - **CNN**: Convolutional Neural Network
    
    Each algorithm has been trained on images preprocessed in one of four ways:
    - **X.P**: Without preprocessing
    - **CLAHE**: Contrast Limited Adaptive Histogram Equalization 
    - **NR**: Noise Reduction
    - **WB**: White Balancing
    
    **Note:** All images are resized to **144 x 144** before processing.
    """)
    with st.expander("About this App"):
        st.write("""
        This app demonstrates how multiple machine learning models can be integrated into a single, interactive GUI.
        After uploading an image of an apple, you can select one of 12 models and click **Start Classification**.
        The app then preprocesses the image as needed, runs inference, and displays the detected apple type along with the detailed classification probabilities.
        """)
elif st.session_state.page == "Detection":
    st.title("Apple Detection")
    st.markdown("Upload an image and choose one of the 12 models to classify the apple type and quality.")
    
    # Upload image file
    uploaded_file = st.file_uploader("Choose an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    
    # Model selection dropdown
    model_option = st.selectbox(
        "Select a model",
        [
            "CNN1 (X.P)", "CNN2 (CLAHE)", "CNN3 (Noise Reduction)", "CNN4 (White Balancing)",
            "MLP1 (X.P)", "MLP2 (CLAHE)", "MLP3 (Noise Reduction)", "MLP4 (White Balancing)",
            "XGB1 (X.P)", "XGB2 (CLAHE)", "XGB3 (Noise Reduction)", "XGB4 (White Balancing)"
        ]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Start Classification"):
            # Convert image to RGB and resize to 144x144
            image_np = np.array(image.convert('RGB'))
            resized_image = cv2.resize(image_np, (144, 144))
            
            # Apply the appropriate preprocessing based on model selection
            preprocess_func = get_preprocessing_func(model_option)
            processed_image = preprocess_func(resized_image)
            
            st.subheader("Processed Image")
            st.image(processed_image, caption="Processed Image", width=300)
            
            # Prepare input based on model type
            if model_option.startswith("CNN"):
                input_data = torch.tensor(processed_image, dtype=torch.float32)
                input_data = input_data.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 144, 144]
            elif model_option.startswith("MLP"):
                input_data = torch.tensor(processed_image, dtype=torch.float32)
                input_data = input_data.view(1, -1)  # Flatten image: [1, 144*144*3]
            elif model_option.startswith("XGB"):
                input_data = processed_image.flatten().reshape(1, -1)
            else:
                input_data = processed_image
            
            # Load the selected model (download if needed)
            model = load_model(model_option)
            st.write("Running inference ...")
            with st.spinner("Classifying..."):
                time.sleep(1)  # Simulate processing delay
                preds = actual_inference(model, input_data)
            
            detected_label = max(preds, key=preds.get)
            confidence = preds[detected_label] * 100
            st.markdown("### Detection Result")
            st.info(f"**Detected Apple Type:** {detected_label} (Confidence: {confidence:.2f}%)")
            
            df_preds = pd.DataFrame(list(preds.items()), columns=["Apple Type", "Probability"]).set_index("Apple Type")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Probability Table")
                st.dataframe(df_preds.style.format("{:.2%}"))
            with col2:
                st.subheader("Probability Chart")
                st.bar_chart(df_preds)
