import sys
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

# Import your model loader from model_file.py
from model_file import load_model

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="GlowGuard – Skin Cancer Risk Assistant",
    page_icon="🩺",
    layout="wide",
)

# -------------------- CUSTOM CSS --------------------
APP_CSS = """
<style>
.main {
    background-color: #050510;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
}
.glowguard-hero {
    font-size: 2.7rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff7aa2, #ffd36b);
    -webkit-background-clip: text;
    color: transparent;
}
.glowguard-sub {
    font-size: 1.05rem;
    color: #cfcfd9;
}
.glowguard-card {
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    background: #14141f;
    border: 1px solid #25253a;
    box-shadow: 0 16px 40px rgba(0,0,0,0.45);
    margin-bottom: 1rem;
}
.glowguard-card-title {
    font-weight: 700;
    font-size: 1.15rem;
    margin-bottom: 0.3rem;
    color: #ffffff;
}
.glowguard-small {
    font-size: 0.9rem;
    color: #b3b3c2;
}
.glowguard-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    background: rgba(255,255,255,0.06);
    color: #e5e5ff;
    margin-right: 6px;
}
.glowguard-section-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 1.0rem;
    margin-bottom: 0.4rem;
    color: #f9f9ff;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# -------------------- SIDEBAR NAVIGATION --------------------
st.sidebar.title("GlowGuard")
st.sidebar.caption("AI skin lesion assistant")

page = st.sidebar.radio(
    "Navigate",
    ["Home feed", "Scan lesion", "Explore cases", "About & safety"],
    index=0,
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def get_model():
    print("GlowGuard: loading model...", file=sys.stderr)
    model = load_model()
    print("GlowGuard: model ready.", file=sys.stderr)
    return model

model = get_model()

# -------------------- HOME FEED --------------------
def show_home_page():
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown('<div class="glowguard-hero">GlowGuard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="glowguard-sub">'
            'A health-focused feed for exploring skin lesions with AI. '
            'Upload images, browse examples, and learn when to see a doctor. '
            '<b>Not a diagnosis.</b>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown('<div class="glowguard-section-title">Quick actions</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="glowguard-card">', unsafe_allow_html=True)
            st.markdown('<div class="glowguard-card-title">📸 Start a Scan</div>', unsafe_allow_html=True)
            st.markdown('<div class="glowguard-small">Upload a lesion image and get an AI risk score.</div>', unsafe_allow_html=True)
            if st.button("Go to Scan ➜", key="go_scan_home"):
                st.session_state["page_override"] = "Scan lesion"
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="glowguard-card">', unsafe_allow_html=True)
            st.markdown('<div class="glowguard-card-title">🎞 Explore Cases</div>', unsafe_allow_html=True)
            st.markdown('<div class="glowguard-small">View example benign & suspicious lesions (demo only).</div>', unsafe_allow_html=True)
            if st.button("Explore ➜", key="go_explore_home"):
                st.session_state["page_override"] = "Explore cases"
            st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glowguard-section-title">About GlowGuard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="glowguard-card glowguard-small">'
            'GlowGuard uses a deep learning model to estimate melanoma risk. '
            'This is an educational tool only.'
            '</div>',
            unsafe_allow_html=True,
        )

# -------------------- SCAN PAGE --------------------
def show_scan_page():
    st.markdown('<div class="glowguard-section-title">Scan a Skin Lesion</div>', unsafe_allow_html=True)
    st.write("Upload a close-up photo to receive an AI-generated risk estimate.")

    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        cimg, cinfo = st.columns([2, 1], gap="large")

        with cimg:
            st.image(image, caption="Uploaded image", use_column_width=True)

        with cinfo:
            st.markdown('<div class="glowguard-card glowguard-small">Keep the lesion centered, sharp, and well-lit.</div>', unsafe_allow_html=True)

        if st.button("Run GlowGuard Scan"):
            with st.spinner("Running model prediction..."):

            
# 🚨🚨 REAL MODEL PREDICTION CODE (from app.copy.py)
# -------------------------------------------------------

# Preprocessing transform
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Apply transforms
x = tf(image).unsqueeze(0)

# Model inference
with torch.no_grad():
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]

# Malignant class = index 1
prob = float(probs[1])   # <-- MUST RETURN VALUE BETWEEN 0 AND 1

# -------------------------------------------------------

            st.success("Scan complete!")

            st.markdown('<div class="glowguard-section-title">Result</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="glowguard-card glowguard-card-title">Estimated malignancy risk: {prob*100:.1f}%</div>', unsafe_allow_html=True)

# -------------------- EXPLORE PAGE --------------------
def show_explore_page():
    st.markdown('<div class="glowguard-section-title">Explore Example Cases</div>', unsafe_allow_html=True)
    st.write("These are demo text-only examples to help you learn patterns.")

    st.markdown('<div class="glowguard-card"><span class="glowguard-pill">Benign</span><div class="glowguard-card-title">Smooth bordered mole</div><div class="glowguard-small">Single color, round, stable over years.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="glowguard-card"><span class="glowguard-pill">Suspicious</span><div class="glowguard-card-title">Irregular, fast-changing lesion</div><div class="glowguard-small">Multiple colors, asymmetry, jagged edges.</div></div>', unsafe_allow_html=True)

# -------------------- ABOUT PAGE --------------------
def show_about_page():
    st.markdown('<div class="glowguard-section-title">About & Safety</div>', unsafe_allow_html=True)
    st.write("""
GlowGuard uses a deep learning model (EfficientNet-B2) to estimate melanoma risk.
This is **not medical advice**.
Always consult a dermatologist for concerning lesions.
""")

# -------------------- ROUTER --------------------
def route(page_name):
    if page_name == "Home feed":
        show_home_page()
    elif page_name == "Scan lesion":
        show_scan_page()
    elif page_name == "Explore cases":
        show_explore_page()
    elif page_name == "About & safety":
        show_about_page()

if "page_override" in st.session_state:
    target = st.session_state.pop("page_override")
    route(target)
else:
    route(page)