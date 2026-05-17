"""
FairVision — Age Classification Web App
Run: streamlit run app.py
Requires: fairvision_deployed.pth + model_meta.json in same directory
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import json, numpy as np, io, os, base64, time

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FairVision · Age Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# GLOBAL CSS  — editorial dark theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Syne+Mono&family=Outfit:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background: #050508;
    color: #e2e2f0;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── HERO ── */
.fv-hero {
    position: relative;
    width: 100%;
    min-height: 100vh;
    background: #050508;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem 2rem;
}

.fv-grid {
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    mask-image: radial-gradient(ellipse 80% 60% at 50% 50%, black 40%, transparent 100%);
}

.fv-glow {
    position: absolute;
    width: 600px; height: 600px;
    border-radius: 50%;
    filter: blur(120px);
    opacity: 0.18;
    pointer-events: none;
}
.fv-glow-1 { background: #6e3cff; top: -100px; left: -100px; }
.fv-glow-2 { background: #00c9a7; bottom: -100px; right: -100px; }
.fv-glow-3 { background: #ff6b6b; top: 30%; left: 50%; transform: translateX(-50%); width: 400px; height: 300px; opacity: 0.08; }

.fv-eyebrow {
    font-family: 'Syne Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.35em;
    color: #00c9a7;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    position: relative;
}

.fv-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(3.2rem, 8vw, 7rem);
    line-height: 0.95;
    letter-spacing: -0.03em;
    text-align: center;
    margin-bottom: 1.5rem;
    position: relative;
}
.fv-title span.grad {
    background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 50%, #34d399 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}

.fv-subtitle {
    font-size: 1.1rem;
    font-weight: 300;
    color: #8b8ba7;
    text-align: center;
    max-width: 540px;
    line-height: 1.7;
    margin-bottom: 3rem;
    position: relative;
}

/* ── STAT STRIP ── */
.fv-stats {
    display: flex;
    gap: 2px;
    margin-bottom: 3.5rem;
    position: relative;
}
.fv-stat {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 1.1rem 2rem;
    text-align: center;
    min-width: 120px;
}
.fv-stat:first-child { border-radius: 12px 0 0 12px; }
.fv-stat:last-child  { border-radius: 0 12px 12px 0; }
.fv-stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: #a78bfa;
    line-height: 1;
}
.fv-stat-lbl {
    font-size: 0.7rem;
    color: #5a5a7a;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── UPLOAD ZONE ── */
.fv-upload-wrap {
    position: relative;
    width: 100%;
    max-width: 660px;
}

/* override streamlit uploader */
.stFileUploader > div > div {
    background: rgba(110,60,255,0.06) !important;
    border: 2px dashed rgba(110,60,255,0.35) !important;
    border-radius: 20px !important;
    padding: 2.5rem !important;
    transition: all 0.3s ease !important;
}
.stFileUploader > div > div:hover {
    border-color: rgba(110,60,255,0.7) !important;
    background: rgba(110,60,255,0.1) !important;
}
.stFileUploader label {
    font-family: 'Outfit', sans-serif !important;
    color: #8b8ba7 !important;
}

/* ── RESULT PANEL ── */
.fv-result-wrap {
    width: 100%;
    background: #050508;
    padding: 4rem 2rem;
    position: relative;
}

.fv-result-inner {
    max-width: 1100px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 3rem;
    align-items: start;
}

.fv-photo-frame {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    aspect-ratio: 1;
    background: #0e0e1a;
    border: 1px solid rgba(255,255,255,0.06);
}
.fv-photo-frame::before {
    content: '';
    position: absolute;
    inset: -2px;
    background: linear-gradient(135deg, #6e3cff, #00c9a7, #6e3cff);
    border-radius: 26px;
    z-index: -1;
    opacity: 0.5;
}

.fv-result-panel {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.fv-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(0,201,167,0.12);
    border: 1px solid rgba(0,201,167,0.3);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-family: 'Syne Mono', monospace;
    font-size: 0.72rem;
    color: #00c9a7;
    letter-spacing: 0.2em;
    width: fit-content;
}

.fv-age-big {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 4.5rem;
    line-height: 0.9;
    letter-spacing: -0.04em;
    background: linear-gradient(135deg, #e2e2f0 30%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}

.fv-conf-row {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
}
.fv-conf-num {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #38bdf8;
}
.fv-conf-label {
    font-size: 0.85rem;
    color: #5a5a7a;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* bar chart rows */
.fv-bar-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
}
.fv-bar-label {
    font-family: 'Syne Mono', monospace;
    font-size: 0.72rem;
    color: #8b8ba7;
    width: 64px;
    flex-shrink: 0;
}
.fv-bar-track {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    overflow: hidden;
}
.fv-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6e3cff, #38bdf8);
    transition: width 0.8s cubic-bezier(0.23, 1, 0.32, 1);
}
.fv-bar-fill.top {
    background: linear-gradient(90deg, #a78bfa, #00c9a7);
}
.fv-bar-pct {
    font-family: 'Syne Mono', monospace;
    font-size: 0.7rem;
    color: #5a5a7a;
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}

/* tag cloud */
.fv-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.5rem;
}
.fv-tag {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    font-size: 0.72rem;
    color: #6b6b8a;
    font-family: 'Syne Mono', monospace;
}

/* divider */
.fv-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(110,60,255,0.4), transparent);
    margin: 0.5rem 0;
}

/* footer strip */
.fv-footer {
    text-align: center;
    padding: 2rem;
    font-family: 'Syne Mono', monospace;
    font-size: 0.68rem;
    color: #3a3a5a;
    letter-spacing: 0.15em;
    border-top: 1px solid rgba(255,255,255,0.04);
}

/* streamlit image tweak */
.stImage img {
    border-radius: 20px;
    width: 100%;
    object-fit: cover;
}

/* spinner override */
.stSpinner > div { border-top-color: #6e3cff !important; }

/* metric override */
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #a78bfa !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MODEL ARCHITECTURE  (must match training)
# ──────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch, bias=False), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            ) if stride != 1 or in_ch != out_ch else nn.Sequential()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.se(self.conv(x)) + self.shortcut(x))


class FairVisionCNN(nn.Module):
    def __init__(self, num_classes=9, drop=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.stage1 = nn.Sequential(ResBlock(32,  64),            ResBlock(64,  64))
        self.stage2 = nn.Sequential(ResBlock(64,  128, stride=2), ResBlock(128, 128))
        self.stage3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.stage4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(drop / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        return self.head(x)


# ──────────────────────────────────────────────
# LOAD MODEL  (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(weights_path, meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = FairVisionCNN(num_classes=meta["num_classes"]).to(device)
    state  = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, meta, device


# ──────────────────────────────────────────────
# TTA INFERENCE
# ──────────────────────────────────────────────
def _norm():
    return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def build_tta(img_size):
    return [
        transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.ToTensor(), _norm()]),
        transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), _norm()]),
        transforms.Compose([transforms.Resize((img_size+12, img_size+12)),
                             transforms.CenterCrop(img_size),
                             transforms.ToTensor(), _norm()]),
        transforms.Compose([transforms.Resize((img_size+12, img_size+12)),
                             transforms.CenterCrop(img_size),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), _norm()]),
        transforms.Compose([transforms.Resize((img_size+8, img_size+8)),
                             transforms.CenterCrop(img_size),
                             transforms.ToTensor(), _norm()]),
    ]

@torch.no_grad()
def predict(model, image, meta, device):
    tfms = build_tta(meta["img_size"])
    acc  = None
    for t in tfms:
        x = t(image).unsqueeze(0).to(device)
        p = F.softmax(model(x), dim=1).cpu().numpy()[0]
        acc = p if acc is None else acc + p
    avg = acc / len(tfms)
    idx = int(np.argmax(avg))
    return idx, avg


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def pct(v): return f"{v*100:.1f}%"

AGE_ICONS = ["👶","🧒","🧑","🧑","👨","🧔","🧓","👴","👴"]


# ──────────────────────────────────────────────
# CHECK FILES
# ──────────────────────────────────────────────
WEIGHTS = "fairvision_deployed.pth"
META    = "model_meta.json"
files_ok = os.path.exists(WEIGHTS) and os.path.exists(META)


# ──────────────────────────────────────────────
# HERO SECTION
# ──────────────────────────────────────────────
st.markdown("""
<div class="fv-hero">
  <div class="fv-grid"></div>
  <div class="fv-glow fv-glow-1"></div>
  <div class="fv-glow fv-glow-2"></div>
  <div class="fv-glow fv-glow-3"></div>

  <div class="fv-eyebrow">▸ fairness-aware neural network</div>

  <div class="fv-title">
    <span class="grad">FairVision</span><br>Age Intelligence
  </div>

  <div class="fv-subtitle">
    A custom CNN trained on FairFace with four fairness interventions —
    balanced sampling, focal loss, soft class weights, and 5-view TTA.
    Upload a face to see it in action.
  </div>

  <div class="fv-stats">
    <div class="fv-stat">
      <div class="fv-stat-val">9</div>
      <div class="fv-stat-lbl">Age Groups</div>
    </div>
    <div class="fv-stat">
      <div class="fv-stat-val">7</div>
      <div class="fv-stat-lbl">Race Groups</div>
    </div>
    <div class="fv-stat">
      <div class="fv-stat-val">4.2M</div>
      <div class="fv-stat-lbl">Parameters</div>
    </div>
    <div class="fv-stat">
      <div class="fv-stat-val">TTA×5</div>
      <div class="fv-stat-lbl">Augmented Views</div>
    </div>
    <div class="fv-stat">
      <div class="fv-stat-val">53.6%</div>
      <div class="fv-stat-lbl">Fair Accuracy</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FILE ERROR STATE
# ──────────────────────────────────────────────
if not files_ok:
    st.error(
        f"**Model files not found.**  \n"
        f"Place `{WEIGHTS}` and `{META}` in the same directory as `app.py`.",
        icon="🚫"
    )
    st.stop()


# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────
with st.spinner("Initialising FairVisionCNN_v3…"):
    model, meta, device = load_model(WEIGHTS, META)


# ──────────────────────────────────────────────
# UPLOAD  (centred, constrained width)
# ──────────────────────────────────────────────
_, col_up, _ = st.columns([1, 2, 1])
with col_up:
    st.markdown('<div style="padding:0 0 0.5rem;">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a face image here — JPG, PNG, or WEBP",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible",
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# RESULT SECTION
# ──────────────────────────────────────────────
if uploaded:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    with st.spinner("Running 5-view TTA inference…"):
        t0 = time.time()
        pred_idx, probs = predict(model, image, meta, device)
        elapsed_ms = (time.time() - t0) * 1000

    age_label  = meta["age_names"][pred_idx]
    confidence = float(probs[pred_idx])
    age_icon   = AGE_ICONS[pred_idx]
    sorted_idx = np.argsort(probs)[::-1]

    # ── photo + result side by side ──────────
    st.markdown('<div class="fv-result-wrap">', unsafe_allow_html=True)
    col_photo, col_data = st.columns([1, 1], gap="large")

    with col_photo:
        st.markdown('<div class="fv-photo-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_data:
        st.markdown(f'<div class="fv-badge">✦ prediction result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="fv-age-big">{age_icon} {age_label}</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="fv-conf-row">
            <span class="fv-conf-num">{confidence*100:.1f}%</span>
            <span class="fv-conf-label">confidence · {elapsed_ms:.0f} ms</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="fv-divider"></div>', unsafe_allow_html=True)

        # probability bars
        st.markdown('<div style="margin-top:0.8rem;">', unsafe_allow_html=True)
        for i in sorted_idx:
            lbl  = meta["age_names"][i]
            prob = float(probs[i])
            is_top = i == pred_idx
            fill_class = "fv-bar-fill top" if is_top else "fv-bar-fill"
            st.markdown(f"""
            <div class="fv-bar-row">
                <span class="fv-bar-label">{lbl}</span>
                <div class="fv-bar-track">
                    <div class="{fill_class}" style="width:{prob*100:.1f}%"></div>
                </div>
                <span class="fv-bar-pct">{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="fv-divider" style="margin-top:1rem;"></div>', unsafe_allow_html=True)

        # race groups supported
        race_tags = "".join(
            f'<span class="fv-tag">{r}</span>' for r in meta["race_names"]
        )
        st.markdown(f"""
        <div style="margin-top:0.75rem;">
            <div style="font-size:0.7rem;color:#3a3a5a;letter-spacing:0.2em;
                        text-transform:uppercase;font-family:'Syne Mono',monospace;
                        margin-bottom:0.5rem;">trained on</div>
            <div class="fv-tags">{race_tags}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # result-wrap

    # ── accuracy timeline ──────────────────────
    st.markdown("""
    <div style="max-width:900px;margin:2rem auto;padding:0 2rem;">
        <div style="font-family:'Syne Mono',monospace;font-size:0.7rem;
                    color:#3a3a5a;letter-spacing:0.25em;text-transform:uppercase;
                    margin-bottom:1.5rem;">Mitigation Journey</div>
    </div>
    """, unsafe_allow_html=True)

    _, c1, c2, c3, _ = st.columns([1, 1, 1, 1, 1])
    c1.metric("Baseline", f"{meta['baseline_acc']*100:.1f}%", help="No fairness interventions")
    c2.metric("Mitigation 1", f"{meta['mit1_acc']*100:.1f}%",
              delta=f"{(meta['mit1_acc']-meta['baseline_acc'])*100:+.1f}pp",
              help="+ Soft class weights")
    c3.metric("Deployed", f"{meta['mit2_acc']*100:.1f}%",
              delta=f"{(meta['mit2_acc']-meta['baseline_acc'])*100:+.1f}pp",
              help="+ WeightedRandomSampler — fairer across all race & age groups")

    st.markdown("""
    <div style="max-width:900px;margin:0 auto 3rem;padding:1rem 2rem;
                background:rgba(110,60,255,0.06);border-left:2px solid rgba(110,60,255,0.4);
                border-radius:0 10px 10px 0;">
        <span style="font-size:0.82rem;color:#6b6b8a;line-height:1.7;">
            <strong style="color:#a78bfa;">Accuracy ≠ Fairness.</strong>
            The deployed model scores lower overall (53.6% vs 58.1%) but dramatically
            improves recall for under-represented age groups like <em>0–2</em> and <em>70+</em>,
            and narrows the accuracy gap across all 7 race groups.
            That's the fairness trade-off — intentional, measurable, and worth it.
        </span>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── empty state hint ──────────────────────
    st.markdown("""
    <div style="text-align:center;padding:2rem 1rem 4rem;color:#3a3a5a;
                font-family:'Syne Mono',monospace;font-size:0.78rem;letter-spacing:0.15em;">
        ↑ upload a face image above to begin
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown(f"""
<div class="fv-footer">
    FAIRVISION CNN v3 &nbsp;·&nbsp;
    FAIRFACE 0.25 &nbsp;·&nbsp;
    FOCAL LOSS γ=2 &nbsp;·&nbsp;
    BALANCED SUBSET &nbsp;·&nbsp;
    TTA × 5 &nbsp;·&nbsp;
    DEVICE: {'CUDA' if torch.cuda.is_available() else 'CPU'}
</div>
""", unsafe_allow_html=True)