import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms

# --------------------------
# LOAD THE MODEL
# --------------------------
@st.cache_resource
def load_model():
    weights_enum = None
    try:
        weights_enum = models.EfficientNet_B2_Weights.IMAGENET1K_V1
    except:
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

model = load_model()

# --------------------------
# STREAMLIT UI
# --------------------------
st.title("Skin Cancer Detection")
st.write("Upload a skin image to analyze risk:")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # preprocessing
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    x = tf(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    benign = float(probs[0] * 100)
    malignant = float(probs[1] * 100)

    st.subheader("Results:")
    st.write(f"Benign: **{benign:.2f}%**")
    st.write(f"Malignant: **{malignant:.2f}%**")

    if malignant >= 70:
        st.error("HIGH RISK — See a dermatologist immediately.")
    elif malignant >= 40:
        st.warning("MEDIUM RISK — Recommended to consult a dermatologist.")
    else:
        st.success("LOW RISK — Looks benign, but monitor changes.")
