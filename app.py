import sys
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="GlowGuard – Skin Cancer Risk Assistant",
    page_icon="🩺",
    layout="wide",
)

# -------------------- LOAD MODEL (from your old app) --------------------
@st.cache_resource
def get_model():
    # This is exactly the same logic you used before
    weights_enum = None
    try:
        weights_enum = models.EfficientNet_B2_Weights.IMAGENET1K_V1
    except Exception:
        weights_enum = None

    # build same architecture
    model = models.efficientnet_b2(weights=weights_enum)

    # replace classifier head to match training
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 2)

    # load trained weights
    state_dict = torch.load("best_model.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    model.eval()
    return model


model = get_model()

# -------------------- CUSTOM CSS (styling) --------------------
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
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff7aa2, #ffd36b);
    -webkit-background-clip: text;
    color: transparent;
}
.glowguard-sub {
    font-size: 1.0rem;
    color: #cfcfd9;
}
.glowguard-card {
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    background: #14141f;
    border: 1px solid #25253a;
    box-shadow: 0 16px 40px rgba(0,0,0,0.45);
    margin-top: 1rem;
}
.glowguard-card-title {
    font-weight: 700;
    font-size: 1.1rem;
    margin-bottom: 0.3rem;
    color: #ffffff;
}
.glowguard-small {
    font-size: 0.9rem;
    color: #b3b3c2;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

# -------------------- MAIN UI --------------------
def main():
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown('<div class="glowguard-hero">GlowGuard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="glowguard-sub">'
            'Upload a skin lesion photo and GlowGuard will estimate the probability that it is malignant. '
            '<b>This is not medical advice.</b>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", use_column_width=True)

            if st.button("Run GlowGuard Scan"):
                with st.spinner("Running model prediction..."):

                    # -------- REAL MODEL PREDICTION CODE --------

                    # Preprocessing transform
                    tf = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225],
                        ),
                    ])

                    # Apply transforms
                    x = tf(image).unsqueeze(0)

                    # Model inference
                    with torch.no_grad():
                        logits = model(x)
                        probs = F.softmax(logits, dim=1)[0]

                    # Malignant probability (class index 1)
                    prob = float(probs[1])  # 0–1

                st.success("Scan complete ✅")

                # --- Interpret prediction as Malignant / Benign + percentages ---
                risk_pct = prob * 100.0
                benign_pct = 100.0 - risk_pct

                # Simple rule: if malignant probability >= 50%, predict malignant
                label = "Malignant" if risk_pct >= 50.0 else "Benign"

                # Main prediction card
                st.markdown('<div class="glowguard-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="glowguard-card-title">Prediction: {label}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    (
                        f'<div class="glowguard-small">'
                        f'Estimated chance this lesion is <b>malignant</b>: {risk_pct:.1f}%<br>'
                        f'Estimated chance this lesion is <b>benign</b>: {benign_pct:.1f}%'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

                # Risk guidance text based on malignant probability
                if risk_pct >= 70:
                    st.markdown(
                        '<div class="glowguard-small">⚠️ This looks like a <b>high-risk</b> lesion based on the model. '
                        'Please see a dermatologist as soon as possible. Only a doctor can diagnose skin cancer.</div>',
                        unsafe_allow_html=True,
                    )
                elif risk_pct >= 40:
                    st.markdown(
                        '<div class="glowguard-small">⚠️ This falls into an <b>intermediate-risk</b> range. '
                        'Consider scheduling an appointment with a dermatologist to have it checked.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="glowguard-small">'
                        'This looks more likely to be <b>benign</b> according to the model, '
                        'but AI can be wrong. If the spot is new, changing, or worrying you, '
                        'still consult a professional.</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown('</div>', unsafe_allow_html=True)

                # Definitions card: malignant vs benign
                st.markdown('<div class="glowguard-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="glowguard-card-title">What do "malignant" and "benign" mean?</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    (
                        '<div class="glowguard-small">'
                        '<b>Benign</b> skin lesions are <b>non-cancerous</b> growths. They do not invade surrounding '
                        'tissues or spread (metastasize) to other parts of the body. Examples include common moles, '
                        'seborrheic keratoses, and many harmless spots that people develop over time.<br><br>'
                        '<b>Malignant</b> skin lesions are <b>cancers</b>, such as melanoma or other skin cancers. '
                        'They can grow into deeper layers of the skin and may spread to lymph nodes or other organs. '
                        'Early detection and treatment are very important.<br><br>'
                        'GlowGuard can only estimate probabilities from a photo. It cannot see beneath the skin, '
                        'take a biopsy, or replace an in-person exam. Always talk to a healthcare professional if you '
                        'notice a new, changing, or worrying lesion.'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("⬆️ Upload a clear close-up lesion photo to get started.")

    with col_right:
        st.markdown('<div class="glowguard-card">', unsafe_allow_html=True)
        st.markdown('<div class="glowguard-card-title">How to take a good photo</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="glowguard-small">'
            '• Center the lesion in the frame<br>'
            '• Make sure it is in focus<br>'
            '• Use good lighting (no heavy flash glare)<br>'
            '• Avoid multiple lesions in one image'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glowguard-card">', unsafe_allow_html=True)
        st.markdown('<div class="glowguard-card-title">Safety note</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="glowguard-small">'
            'GlowGuard is an educational project, not a medical device. '
            'Do not use it to make medical decisions. Always consult a doctor for any concerns.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
