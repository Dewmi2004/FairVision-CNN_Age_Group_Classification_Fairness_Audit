import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json, numpy as np, io, os

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title="FairVision",
    page_icon="👁",
    layout="centered"
)

st.title("👁 FairVision")
st.caption("Simple Face Intelligence System")

# ─────────────────────────────────────
# MODEL
# ─────────────────────────────────────
class FairVisionCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ─────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model_meta.json") as f:
        meta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FairVisionCNN(num_classes=meta["num_classes"]).to(device)
    model.load_state_dict(torch.load("fairvision_deployed.pth", map_location=device))
    model.eval()

    return model, meta, device

# ─────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────
def transform_image(img, size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img).unsqueeze(0)

# ─────────────────────────────────────
# PREDICT
# ─────────────────────────────────────
def predict(model, img, meta, device):
    x = transform_image(img, meta["img_size"]).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]

    idx = int(np.argmax(probs))
    return meta["age_names"][idx], probs[idx]

# ─────────────────────────────────────
# FILE CHECK
# ─────────────────────────────────────
if not os.path.exists("fairvision_deployed.pth") or not os.path.exists("model_meta.json"):
    st.error("❌ Model files not found")
    st.stop()

model, meta, device = load_model()

# ─────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────
uploaded = st.file_uploader("📁 Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    # 🔽 Reduce image size (IMPORTANT)
    image = image.resize((300, 300))

    st.image(image, caption="Uploaded Image", width=250)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing..."):
            age, conf = predict(model, image, meta, device)

        st.success("Analysis Complete")

        # ─────────────────────────────
        # RESULTS
        # ─────────────────────────────
        st.subheader("Result")

        st.metric("Age Group", age)
        st.metric("Confidence", f"{conf*100:.2f}%")

        # Dummy values (replace if needed)
        st.metric("Gender", "Male")
        st.metric("Ethnicity", "Asian")

else:
    st.info("Upload an image to start")

# ─────────────────────────────────────
# FOOTER
# ─────────────────────────────────────
st.markdown("---")
st.caption("FairVision · Simple Version · AI Model")
