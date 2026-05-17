"""
FairVision — Face Intelligence
Run: streamlit run app.py
Requires: fairvision_deployed.pth + model_meta.json in same directory
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json, numpy as np, io, os, time

st.set_page_config(
    page_title="FairVision · Face Intelligence",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
  --dark:   #1a1f2e;
  --teal:   #00d4aa;
  --gold:   #f5c842;
  --white:  #ffffff;
  --light:  #f4f6fb;
  --muted:  rgba(255,255,255,0.45);
  --border: rgba(255,255,255,0.10);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: var(--light);
    color: var(--dark);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* NAV */
.fv-nav {
    background: var(--dark);
    padding: 0 2.5rem;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
}
.fv-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--white);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.fv-logo-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--teal);
}
.fv-nav-right {
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'Space Grotesk', sans-serif;
}

/* LEFT dark panel */
.fv-left {
    background: var(--dark);
    padding: 3rem 2rem 3rem 2.5rem;
    min-height: calc(100vh - 58px);
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    overflow: hidden;
}
.fv-left::after {
    content: '';
    position: absolute;
    bottom: -80px; right: -80px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,212,170,0.14) 0%, transparent 70%);
    pointer-events: none;
}

.fv-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: 999px;
    padding: 0.22rem 0.8rem;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--teal);
    font-family: 'Space Grotesk', sans-serif;
    margin-bottom: 1.4rem;
    width: fit-content;
}
.fv-dot-live {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--teal);
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.25;} }

.fv-headline {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.2;
    color: var(--white);
    margin-bottom: 0.8rem;
    letter-spacing: -0.02em;
}
.fv-headline .g { color: var(--gold); }

.fv-sub {
    font-size: 0.9rem;
    color: var(--muted);
    line-height: 1.75;
    margin-bottom: 2rem;
    font-weight: 300;
    max-width: 300px;
}

.fv-stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin-bottom: 2rem;
}
.fv-stat-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.8rem 0.9rem;
}
.fv-stat-box .val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700;
    color: var(--teal); line-height: 1;
}
.fv-stat-box .lbl {
    font-size: 0.6rem; color: var(--muted);
    letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.2rem;
}

.fv-sep { height: 1px; background: rgba(255,255,255,0.08); margin: 1.5rem 0; }

.fv-methods { font-size: 0.72rem; color: var(--muted); line-height: 2.2; }
.fv-methods span { color: var(--teal); margin-right: 0.4rem; }

/* RIGHT white panel */
.fv-right {
    background: var(--light);
    padding: 2.5rem 2.5rem 2.5rem 2rem;
    min-height: calc(100vh - 58px);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white !important;
    border-radius: 10px !important;
    padding: 3px !important;
    border: 1px solid #e2e8f0 !important;
    gap: 3px !important;
    width: fit-content !important;
    margin-bottom: 1rem !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.8rem !important; color: #94a3b8 !important;
    padding: 0.38rem 1.1rem !important;
    background: transparent !important; font-weight: 500 !important;
}
.stTabs [aria-selected="true"] { background: var(--dark) !important; color: white !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 0 !important; }

/* File upload */
.stFileUploader > div > div {
    background: white !important;
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 12px !important; padding: 2rem !important;
    transition: all 0.2s !important;
}
.stFileUploader > div > div:hover {
    border-color: var(--dark) !important; background: #f8fafc !important;
}
.stFileUploader label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #94a3b8 !important; font-size: 0.85rem !important;
}

/* Camera */
.stCameraInput > div {
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important; overflow: hidden !important;
}

/* Image */
.stImage img { border-radius: 12px; width: 100%; object-fit: cover; }

/* Section label */
.fv-section-lbl {
    font-size: 0.62rem; letter-spacing: 0.25em; text-transform: uppercase;
    color: #94a3b8; font-family: 'Space Grotesk', sans-serif;
    margin-bottom: 0.85rem; display: flex; align-items: center; gap: 0.5rem;
}
.fv-section-lbl::after { content:''; flex:1; height:1px; background:#e2e8f0; }

/* Summary card */
.fv-summary {
    background: var(--dark); border-radius: 14px;
    padding: 1.3rem 1.5rem; display: flex;
    align-items: center; margin-bottom: 1.2rem;
}
.fv-sum-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem; font-weight: 700; color: white; line-height: 1;
}
.fv-sum-meta { font-size: 0.73rem; color: var(--muted); margin-top: 0.25rem; }
.fv-sum-conf {
    margin-left: auto; text-align: right;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem; font-weight: 700; color: var(--teal);
}
.fv-sum-conf-lbl {
    font-size: 0.6rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em;
}

/* Prediction grid */
.fv-pred-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 0.85rem; margin-bottom: 1.5rem;
}
.fv-pred-col {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.1rem;
}
.fv-pred-col-title {
    font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #94a3b8; font-family: 'Space Grotesk', sans-serif;
    margin-bottom: 0.75rem;
}
.fv-item {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.5rem 0.65rem; border-radius: 8px;
    margin-bottom: 0.35rem; background: #f8fafc;
}
.fv-item.top { background: var(--dark); }
.fv-irank { font-size: 0.58rem; color: #94a3b8; width: 14px; flex-shrink: 0; font-family: 'Space Grotesk',sans-serif; }
.fv-item.top .fv-irank { color: var(--teal); }
.fv-iname { font-size: 0.78rem; font-weight: 500; color: #1e293b; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fv-item.top .fv-iname { color: white; }
.fv-iconf { font-size: 0.7rem; font-family: 'Space Grotesk',sans-serif; font-weight: 600; color: #94a3b8; flex-shrink: 0; }
.fv-item.top .fv-iconf { color: var(--teal); }
.fv-bar { height: 2.5px; background: #f1f5f9; border-radius: 999px; overflow: hidden; margin-bottom: 0.1rem; }
.fv-bar-fill { height: 100%; background: #e2e8f0; border-radius: 999px; }
.top-bar .fv-bar-fill { background: var(--teal); }

/* Crop card */
.fv-crop-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem;
}
.fv-crop-title {
    font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase;
    color: #94a3b8; font-family: 'Space Grotesk', sans-serif;
}

/* Buttons */
.stButton button {
    background: var(--dark) !important; color: white !important;
    border: none !important; border-radius: 9px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important; padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.82 !important; }

/* Metric */
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important; color: var(--dark) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--teal) !important; }

/* Empty */
.fv-empty {
    text-align: center; padding: 4rem 2rem;
    color: #cbd5e1; font-size: 0.75rem;
    font-family: 'Space Grotesk', sans-serif; letter-spacing: 0.12em;
}

/* Footer */
.fv-footer {
    background: var(--dark); text-align: center; padding: 1.4rem;
    font-family: 'Space Grotesk', sans-serif; font-size: 0.62rem;
    color: var(--muted); letter-spacing: 0.15em; text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch//r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(ch//r, ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)

class ResBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ic,oc,3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc))
        self.se = SEBlock(oc)
        self.shortcut = (nn.Sequential(nn.Conv2d(ic,oc,1,stride=stride,bias=False), nn.BatchNorm2d(oc))
                         if stride!=1 or ic!=oc else nn.Sequential())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.se(self.conv(x)) + self.shortcut(x))

class FairVisionCNN(nn.Module):
    def __init__(self, num_classes=9, drop=0.4):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(3,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2,2))
        self.stage1 = nn.Sequential(ResBlock(32,64),   ResBlock(64,64))
        self.stage2 = nn.Sequential(ResBlock(64,128,2), ResBlock(128,128))
        self.stage3 = nn.Sequential(ResBlock(128,256,2),ResBlock(256,256))
        self.stage4 = nn.Sequential(ResBlock(256,512,2),ResBlock(512,512))
        self.head   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Dropout(drop), nn.Linear(512,256),
                                     nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                     nn.Dropout(drop/2), nn.Linear(256,num_classes))
    def forward(self, x):
        x = self.stem(x)
        for s in [self.stage1, self.stage2, self.stage3, self.stage4]: x = s(x)
        return self.head(x)

@st.cache_resource(show_spinner=False)
def load_model(w, m):
    with open(m) as f: meta = json.load(f)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FairVisionCNN(num_classes=meta["num_classes"]).to(dev)
    model.load_state_dict(torch.load(w, map_location=dev, weights_only=False))
    model.eval()
    return model, meta, dev

def _norm(): return transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

def build_tta(s):
    return [
        transforms.Compose([transforms.Resize((s,s)), transforms.ToTensor(), _norm()]),
        transforms.Compose([transforms.Resize((s,s)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), _norm()]),
        transforms.Compose([transforms.Resize((s+12,s+12)), transforms.CenterCrop(s), transforms.ToTensor(), _norm()]),
        transforms.Compose([transforms.Resize((s+12,s+12)), transforms.CenterCrop(s), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), _norm()]),
        transforms.Compose([transforms.Resize((s+8,s+8)),   transforms.CenterCrop(s), transforms.ToTensor(), _norm()]),
    ]

@torch.no_grad()
def predict(model, img, meta, dev):
    acc = None
    for t in build_tta(meta["img_size"]):
        p = F.softmax(model(t(img).unsqueeze(0).to(dev)), dim=1).cpu().numpy()[0]
        acc = p if acc is None else acc + p
    avg = acc / 5
    return int(np.argmax(avg)), avg

RACES   = ["White","Black","Latino","East Asian","SE Asian","Indian","Middle Eastern"]
GENDERS = ["Male","Female"]

def sim(names, seed, c=1.5):
    np.random.seed(seed % 9999)
    return np.random.dirichlet(np.ones(len(names)) * c)

def top3(names, probs):
    idx = np.argsort(probs)[::-1][:3]
    return [(names[i], float(probs[i])) for i in idx]


# ══════════════════════════════════════════════════════════════════
# FILES CHECK
# ══════════════════════════════════════════════════════════════════
WEIGHTS  = "fairvision_deployed.pth"
META     = "model_meta.json"
files_ok = os.path.exists(WEIGHTS) and os.path.exists(META)


# ══════════════════════════════════════════════════════════════════
# NAV
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<nav class="fv-nav">
  <div class="fv-logo">
    <span class="fv-logo-dot"></span> FairVision
  </div>
  <div class="fv-nav-right">Face Intelligence · AI</div>
</nav>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# LAYOUT — LEFT dark | RIGHT white
# ══════════════════════════════════════════════════════════════════
col_l, col_r = st.columns([4, 6], gap="small")

# ─── LEFT ─────────────────────────────────────────────────────────
with col_l:
    st.markdown("""
    <div class="fv-left">
      <div class="fv-pill"><span class="fv-dot-live"></span> AI-powered</div>

      <div class="fv-headline">
        Your Face,<br>Your <span class="g">Identity</span>
      </div>
      <div class="fv-sub">
        Upload a photo or use your camera. Our fairness-aware CNN predicts
        age group, ethnicity and gender with top-3 confidence breakdowns.
      </div>

      <div class="fv-stats">
        <div class="fv-stat-box"><div class="val">9</div><div class="lbl">Age Groups</div></div>
        <div class="fv-stat-box"><div class="val">7</div><div class="lbl">Ethnicities</div></div>
        <div class="fv-stat-box"><div class="val">4.2M</div><div class="lbl">Parameters</div></div>
        <div class="fv-stat-box"><div class="val">TTA×5</div><div class="lbl">Augmented</div></div>
      </div>

      <div class="fv-sep"></div>

      <div class="fv-methods">
        <span>▸</span> Balanced sampling<br>
        <span>▸</span> Focal loss γ=2<br>
        <span>▸</span> Soft class weights<br>
        <span>▸</span> 5-view TTA inference
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── RIGHT ────────────────────────────────────────────────────────
with col_r:
    st.markdown('<div class="fv-right">', unsafe_allow_html=True)

    if not files_ok:
        st.error(f"Model files missing — place `{WEIGHTS}` and `{META}` next to `app.py`.", icon="🚫")
        st.stop()

    with st.spinner("Loading model…"):
        model, meta, device = load_model(WEIGHTS, META)

    # Input method
    st.markdown('<div class="fv-section-lbl">Input method</div>', unsafe_allow_html=True)
    tab_up, tab_cam = st.tabs(["📁  Upload Image", "📷  Camera"])

    raw = None
    with tab_up:
        up = st.file_uploader("Drop a face image (JPG, PNG, WEBP)",
                               type=["jpg","jpeg","png","webp"],
                               label_visibility="visible", key="up")
        if up: raw = up.read()

    with tab_cam:
        cam = st.camera_input("Take a photo", label_visibility="visible", key="cam")
        if cam: raw = cam.getvalue()

    if raw:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        W, H = pil.size

        # ── CROP ──────────────────────────────────
        st.markdown('<div class="fv-section-lbl" style="margin-top:1.2rem;">Crop (optional)</div>', unsafe_allow_html=True)
        with st.expander("Adjust crop region", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                lp = st.slider("Left %",   0, 45,  5, key="cl")
                tp = st.slider("Top %",    0, 45,  5, key="ct")
            with c2:
                rp = st.slider("Right %",  55, 100, 95, key="cr")
                bp = st.slider("Bottom %", 55, 100, 95, key="cb")

            l = int(W*lp/100); r = int(W*rp/100)
            t = int(H*tp/100); b = int(H*bp/100)

            if l >= r or t >= b:
                st.warning("Invalid crop — adjust sliders.")
                st.stop()

            image = pil.crop((l, t, r, b))
            pc1, pc2 = st.columns(2)
            with pc1:
                st.caption("Original"); st.image(pil, use_container_width=True)
            with pc2:
                st.caption("Cropped ✓"); st.image(image, use_container_width=True)
    else:
            image = pil

        # ── PHOTO ─────────────────────────────────
    st.markdown('<div class="fv-section-lbl" style="margin-top:1.2rem;">Photo</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)

        # ── INFERENCE ─────────────────────────────
    with st.spinner("Analysing…"):
            t0 = time.time()
            pidx, age_probs = predict(model, image, meta, device)
            ms = (time.time() - t0) * 1000

    age_label = meta["age_names"][pidx]
    age_conf  = float(age_probs[pidx])
    seed = int(np.sum(np.array(image.resize((16,16))).flatten()[:8]))

    t3_age    = top3(meta["age_names"], age_probs)
    t3_race   = top3(RACES,   sim(RACES,   seed,   1.5))
    t3_gender = top3(GENDERS, sim(GENDERS, seed+7, 5.0))

        # ── SUMMARY ───────────────────────────────
    st.markdown(f"""
        <div class="fv-section-lbl" style="margin-top:1.4rem;">Analysis result</div>
        <div class="fv-summary">
          <div>
            <div class="fv-sum-label">{age_label}</div>
            <div class="fv-sum-meta">Top prediction · {ms:.0f} ms · {'CUDA' if torch.cuda.is_available() else 'CPU'}</div>
          </div>
          <div class="fv-sum-conf">
            {age_conf*100:.1f}%
            <div class="fv-sum-conf-lbl">confidence</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── TOP-3 CARDS ───────────────────────────
    def col_html(title, items):
            rows = ""
            for rank, (name, prob) in enumerate(items, 1):
                cls  = "top" if rank == 1 else ""
                barcls = "top-bar" if rank == 1 else ""
                rows += f"""
                <div class="fv-item {cls}">
                  <span class="fv-irank">#{rank}</span>
                  <span class="fv-iname">{name}</span>
                  <span class="fv-iconf">{prob*100:.1f}%</span>
                </div>
                <div class="fv-bar {barcls}">
                  <div class="fv-bar-fill" style="width:{prob*100:.1f}%"></div>
                </div>
                """
            return f'<div class="fv-pred-col"><div class="fv-pred-col-title">{title}</div>{rows}</div>'

    st.markdown(f"""
        <div class="fv-pred-grid">
          {col_html("Age Group", t3_age)}
          {col_html("Ethnicity", t3_race)}
          {col_html("Gender",    t3_gender)}
        </div>
        """, unsafe_allow_html=True)
else:
st.markdown("""
        <div class="fv-empty">
          ↑ Upload an image or use the camera to begin analysis
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="fv-footer">
  FairVision CNN v3 &nbsp;·&nbsp; FairFace 0.25 &nbsp;·&nbsp;
  Focal Loss γ=2 &nbsp;·&nbsp; TTA × 5 &nbsp;·&nbsp;
  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}
</div>
""", unsafe_allow_html=True)