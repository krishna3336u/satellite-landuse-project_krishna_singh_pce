import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Satellite Change Detector", layout="wide")

st.title("Satellite Image Land-Use Classifier & Change Detector")

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "checkpoints/resnet18_best_final.pt"
CLASS_TO_IDX_PATH = "metadata/class_to_idx.json"
IDX_TO_CLASS_PATH = "metadata/idx_to_class.json"
CHANGE_SUMMARY_PATH = "metadata/change_detection_summary.json"

# -----------------------------
# Load metadata
# -----------------------------
with open(CLASS_TO_IDX_PATH, "r") as f:
    class_to_idx = json.load(f)

with open(IDX_TO_CLASS_PATH, "r") as f:
    idx_to_class = json.load(f)

with open(CHANGE_SUMMARY_PATH, "r") as f:
    change_summary = json.load(f)

SIMILARITY_THRESHOLD = change_summary["selected_similarity_threshold"]
NUM_CLASSES = len(class_to_idx)

# idx_to_class keys come as strings from json
idx_to_class = {int(k): v for k, v in idx_to_class.items()}

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Image transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Load classification model
# -----------------------------
@st.cache_resource
def load_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------------
# Create feature extractor
# -----------------------------
@st.cache_resource
def get_feature_extractor():
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor

feature_extractor = get_feature_extractor()

# -----------------------------
# Helper functions
# -----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    return img_tensor

def predict_class(image):
    img_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

    pred_idx = pred_idx.item()
    confidence = confidence.item()
    pred_class = idx_to_class[pred_idx]

    return pred_class, confidence, probs.cpu().numpy()[0]

def extract_embedding(image):
    img_tensor = preprocess_image(image)

    with torch.no_grad():
        embedding = feature_extractor(img_tensor)   # [1, 512, 1, 1]
        embedding = embedding.view(embedding.size(0), -1)   # [1, 512]

    return embedding

def compute_similarity(emb1, emb2):
    similarity = F.cosine_similarity(emb1, emb2).item()
    return similarity

def create_change_heatmap(img1, img2):
    img1 = img1.resize((224, 224))
    img2 = img2.resize((224, 224))

    img1_np = np.array(img1).astype(np.float32)
    img2_np = np.array(img2).astype(np.float32)

    diff = np.abs(img1_np - img2_np)
    heatmap = diff.mean(axis=2)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def overlay_heatmap_on_image(image, heatmap):
    image = image.resize((224, 224))
    image_np = np.array(image)

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
    return overlay

# -----------------------------
# File upload
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    before_file = st.file_uploader("Upload BEFORE image (T1)", type=["jpg", "jpeg", "png", "tif"], key="before")

with col2:
    after_file = st.file_uploader("Upload AFTER image (T2)", type=["jpg", "jpeg", "png", "tif"], key="after")

# -----------------------------
# Prediction section
# -----------------------------
if before_file is not None and after_file is not None:
    before_image = Image.open(before_file).convert("RGB")
    after_image = Image.open(after_file).convert("RGB")

    # Predict classes
    before_class, before_conf, before_probs = predict_class(before_image)
    after_class, after_conf, after_probs = predict_class(after_image)

    # Extract embeddings
    before_emb = extract_embedding(before_image)
    after_emb = extract_embedding(after_image)

    # Similarity
    similarity = compute_similarity(before_emb, after_emb)

    # Change decision
    change_flag = similarity < SIMILARITY_THRESHOLD

    # Heatmap
    heatmap = create_change_heatmap(before_image, after_image)
    overlay_before = overlay_heatmap_on_image(before_image, heatmap)
    overlay_after = overlay_heatmap_on_image(after_image, heatmap)

    st.subheader("Prediction Results")

    c1, c2 = st.columns(2)

    with c1:
        st.image(before_image, caption="Before Image (T1)", use_container_width=True)
        st.write(f"**Predicted Class:** {before_class}")
        st.write(f"**Confidence:** {before_conf:.4f}")

    with c2:
        st.image(after_image, caption="After Image (T2)", use_container_width=True)
        st.write(f"**Predicted Class:** {after_class}")
        st.write(f"**Confidence:** {after_conf:.4f}")

    st.markdown("---")
    st.subheader("Change Detection")

    st.write(f"**Cosine Similarity Score:** {similarity:.4f}")
    st.write(f"**Selected Threshold:** {SIMILARITY_THRESHOLD:.4f}")

    if change_flag:
        st.error("Change Detected")
    else:
        st.success("No Significant Change Detected")

    st.markdown("---")
    st.subheader("Heatmap Visualization")

    h1, h2 = st.columns(2)

    with h1:
        st.image(overlay_before, caption="Before Image + Heatmap Overlay", use_container_width=True)

    with h2:
        st.image(overlay_after, caption="After Image + Heatmap Overlay", use_container_width=True)

    st.markdown("---")
    st.subheader("Raw Difference Heatmap")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(heatmap, cmap="hot")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("Please upload both BEFORE and AFTER images to start.")