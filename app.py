import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="FairVision - Age Classification",
    page_icon="🧠",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #FFFFFF;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #AAAAAA;
    margin-bottom: 30px;
}
.card {
    background-color: #1C1F26;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}
.result {
    font-size: 24px;
    font-weight: bold;
    color: #00FFAA;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="title">FairVision 🧠</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Age Group Classification & Fairness Audit</div>', unsafe_allow_html=True)

# ------------------ MODEL LOAD ------------------
MODEL_PATH = "fairvision_deployed.pth"

@st.cache_resource
def load_model():
    try:
        # 🔥 Try loading as FULL model
        model = torch.load(MODEL_PATH, map_location="cpu")
        model.eval()
        return model
    except:
        # 🔥 If failed → assume state_dict with ResNet
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)

        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)

        model.eval()
        return model

model = load_model()

# ------------------ IMAGE TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------ UPLOAD SECTION ------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PREDICTION ------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    classes = ["Young", "Adult", "Old"]
    result = classes[predicted.item()]

    st.markdown(f'<div class="card result">Prediction: {result}</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<hr style="margin-top:40px">
<p style='text-align:center; color:gray;'>
Built with ❤️ using Streamlit | FairVision Project
</p>
""", unsafe_allow_html=True)
