import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
import xgboost 
from PIL import Image
from streamlit_option_menu import option_menu
import os
import gdown
import io
from rembg import remove  # pip install rembg

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Apple Quality & Type Classification",
    page_icon="🍎",
    layout="wide"
)

# ---------------------------
# Sidebar: Navigation and Settings
# ---------------------------
with st.sidebar:
    st.title("Navigation")
    selected = option_menu(
        menu_title="Pages",
        options=["Home", "Classification"],
        icons=["house", "kanban"],
        menu_icon="cast",
        default_index=0,
    )
    st.markdown("---")
    st.header("Display Settings")
    img_size = st.slider("Select Image Display Size (px)", 200, 500, 300)
    st.info("Adjust the display size of images.")

    st.markdown("---")
    st.info("To further customize the look of your app, check out [Streamlit Themes](https://docs.streamlit.io/library/advanced-features/theming) and update your configuration accordingly.")

# ---------------------------
# Global Variables and Class Names
# ---------------------------
class_names = ['barnae_apple', 'crimson_snow_apples', 'fuji_apple', 
               'gala_apple', 'green_apple', 'pink_lady_apple', 
               'red_delicious_apple', 'rotten_Apple']

# ---------------------------
# Preprocessing Functions
# ---------------------------
def no_preprocessing(image):
    return image

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def apply_white_balance(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    avg_a = np.mean(a)
    avg_b = np.mean(b)
    a = a - (avg_a - 128)
    b = b - (avg_b - 128)
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)
    balanced_lab = cv2.merge((l, a, b))
    return cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2RGB)

def apply_noise_reduction(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

preprocessing_functions = {
    "X.P": no_preprocessing,
    "CLAHE": apply_clahe,
    "Noise Reduction": apply_noise_reduction,
    "White Balancing": apply_white_balance,
}

# ---------------------------
# Torch Model Architectures
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 9 * 9, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# Model Files, IDs, and Architecture Mapping
# ---------------------------
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

model_architectures = {
    "CNN1 (X.P)": "CNN",
    "CNN2 (CLAHE)": "CNN",
    "CNN3 (Noise Reduction)": "CNN",
    "CNN4 (White Balancing)": "CNN",
    "MLP1 (X.P)": "MLP",
    "MLP2 (CLAHE)": "MLP",
    "MLP3 (Noise Reduction)": "MLP",
    "MLP4 (White Balancing)": "MLP",
    "XGB1 (X.P)": "XGB",
    "XGB2 (CLAHE)": "XGB",
    "XGB3 (Noise Reduction)": "XGB",
    "XGB4 (White Balancing)": "XGB"
}

# ---------------------------
# Function to Download Model if Needed
# ---------------------------
def download_model(model_key):
    model_path = model_file_names[model_key]
    if not os.path.exists(model_path):
        file_id = model_file_ids[model_key]
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info(f"Downloading {model_key} from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    return model_path

# ---------------------------
# Helper Functions for Loading and Prediction
# ---------------------------
def get_preprocessing_function(model_key):
    start = model_key.find("(")
    end = model_key.find(")")
    if start != -1 and end != -1:
        preprocess_type = model_key[start+1:end].strip()
        return preprocessing_functions.get(preprocess_type, no_preprocessing)
    return no_preprocessing

@st.cache_resource(show_spinner=False)
def load_torch_model(model_key):
    download_model(model_key)
    arch = model_architectures.get(model_key, "MLP")
    if arch == "CNN":
        model = CNN(num_classes=8)
    elif arch == "MLP":
        input_size = 144 * 144 * 3
        model = MLP(input_size, 8)
    else:
        raise ValueError("Unknown architecture for model key: " + model_key)
    model_path = model_file_names[model_key]
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_xgb_model(model_key):
    download_model(model_key)
    model_path = model_file_names[model_key]
    return joblib.load(model_path)

def predict_image(model_key, image_np):
    image_resized = cv2.resize(image_np, (144, 144))
    preprocess_func = get_preprocessing_function(model_key)
    processed_image = preprocess_func(image_resized)
    processed_image_norm = processed_image / 255.0

    if model_key.startswith("XGB"):
        image_flat = processed_image_norm.flatten().reshape(1, -1)
        model = load_xgb_model(model_key)
        prediction = model.predict(image_flat)[0]
    else:
        arch = model_architectures.get(model_key, "MLP")
        if arch == "MLP":
            image_tensor = torch.tensor(processed_image_norm, dtype=torch.float32).view(1, -1)
        else:
            image_tensor = torch.tensor(processed_image_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        model = load_torch_model(model_key)
        with torch.no_grad():
            output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return prediction, processed_image

# ---------------------------
# Functions for Background Removal
# ---------------------------
def remove_background_transparent(pil_img):
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    output_bytes = remove(img_bytes)
    transparent_image = Image.open(io.BytesIO(output_bytes))
    return transparent_image

def fill_transparency(image, background_color=(255,255,255)):
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, background_color)
        background.paste(image, mask=image.split()[3])
        return background
    else:
        return image

# ---------------------------
# Streamlit App: Home and Classification Pages
# ---------------------------
if selected == "Home":
    st.title("Apple Quality & Type Classification")
    
    # Container with detailed instructions
    with st.container():
        st.header("Welcome!")
        st.markdown("""
        **How to Use This App:**
        1. **Navigate:** Use the sidebar to switch between *Home* and *Classification*.
        2. **Learn:** Read through the instructions below to understand how the app works.
        3. **Upload:** On the Classification page, upload an image of an apple.
        4. **Select a Model:** Choose from several pre-trained models that use different preprocessing techniques.
        5. **Classify:** Click **Start Classify** to see the preprocessed image and predicted apple type.
        """)
    
    with st.expander("Learn More About Apple Classes & Preprocessing"):
        st.write(
            """
            **Apple Classes:**
            - barnae_apple  
            - crimson_snow_apples  
            - fuji_apple  
            - gala_apple  
            - green_apple  
            - pink_lady_apple  
            - red_delicious_apple  
            - rotten_Apple  

            **Preprocessing Techniques:**
            - **X.P:** No additional processing.
            - **CLAHE:** Adaptive histogram equalization.
            - **Noise Reduction:** Bilateral filtering.
            - **White Balancing:** Color correction in LAB space.
            """
        )

elif selected == "Classification":
    st.title("Apple Classification")
    
    # Container with usage instructions and model details side-by-side
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("How to Classify:")
            st.markdown("""
            - **Step 1:** Upload your image in the *Image Upload & Preview* tab.
            - **Step 2:** The app will display your original image and a version with the background removed.
            - **Step 3:** Choose a model from the dropdown menu.
            - **Step 4:** Click **Start Classify** to process the image and get a prediction.
            """)
        with col2:
            st.subheader("Selected Model Details:")
            model_key = st.selectbox("Select Model", list(model_file_names.keys()))
            with st.expander("Model Details"):
                arch = model_architectures.get(model_key, "N/A")
                preprocess_func = get_preprocessing_function(model_key).__name__
                st.markdown(f"**Architecture:** {arch}")
                st.markdown(f"**Preprocessing Technique:** {preprocess_func}")
    
    # Tabs for image upload/preview and classification result.
    tab1, tab2 = st.tabs(["Image Upload & Preview", "Classification Result"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, or png)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGBA")
            image_display = image.resize((img_size, img_size))
            apple_transparent = remove_background_transparent(image)
            apple_transparent_display = apple_transparent.resize((img_size, img_size))
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(image_display, caption="Uploaded Image", width=img_size)
            with col_img2:
                st.image(apple_transparent_display, caption="Image with Transparent Background", width=img_size)
    
    if uploaded_file is not None and st.button("Start Classify"):
        with st.spinner("Classifying..."):
            apple_for_classification = fill_transparency(apple_transparent)
            image_np = np.array(apple_for_classification)
            prediction_idx, processed_image = predict_image(model_key, image_np)
        processed_display = cv2.resize(processed_image, (img_size, img_size))
        with tab2:
            st.image(processed_display, caption="Preprocessed Image", width=img_size)
            st.header("Predicted Class: " + class_names[prediction_idx])
            st.success("Classification complete!")
