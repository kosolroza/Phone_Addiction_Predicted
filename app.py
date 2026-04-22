import streamlit as st
import numpy as np
import pickle
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smartphone Addiction Predictor",
    page_icon="📱",
    layout="centered"
)

# ── Load model ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    theta  = model["theta"]
    x_mean = model["x_mean"]
    x_std  = model["x_std"]
except FileNotFoundError:
    st.error("⚠️  `model.pkl` not found.  Place it in the same folder as this script.")
    st.stop()

# ── Sigmoid helper ─────────────────────────────────────────────────────────────
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📱 Smartphone Addiction Predictor")
st.markdown("Answer the questions below honestly. The model will predict whether you show signs of smartphone addiction.")
st.divider()

# Feature 1 – hours_spend_per_day
hours_label = st.selectbox(
    "1. How many hours do you spend on your smartphone per day?",
    options=["1-2 hours", "3-4 hours", "5 or more hours"]
)
hours_map = {"1-2 hours": 1, "3-4 hours": 2, "5 or more hours": 3}

# Feature 2 – class_use
class_label = st.selectbox(
    "2. How often do you use your smartphone during class?",
    options=["Rarely", "Occasionally", "Frequently", "Always"]
)
class_map = {"Rarely": 0, "Occasionally": 1, "Frequently": 2, "Always": 3}

# Feature 3 – entertainment
entertainment_label = st.selectbox(
    "3. How many hours per day do you use your phone for entertainment (social media, games, videos)?",
    options=["Less than 1 hour", "1-2 hours", "3-4 hours", "5 or more hours"]
)
entertainment_map = {"Less than 1 hour": 0, "1-2 hours": 1, "3-4 hours": 2, "5 or more hours": 3}

# Feature 4 – dependency
dependency_label = st.selectbox(
    "4. How much do you feel dependent on your smartphone?",
    options=["Not at all", "Slightly", "Moderately", "Very much", "Absolutely"]
)
dependency_map = {"Not at all": 0, "Slightly": 1, "Moderately": 2, "Very much": 3, "Absolutely": 4}

# Feature 5 – check_without_noti
check_noti_label = st.selectbox(
    "5. How often do you check your phone without any notification?",
    options=["Never", "Rarely", "Occasionally", "Frequently", "Always"]
)
check_noti_map = {"Never": 0, "Rarely": 1, "Occasionally": 2, "Frequently": 3, "Always": 4}

# Feature 6 – before_bed
before_bed_label = st.selectbox(
    "6. How often do you use your smartphone right before going to bed?",
    options=["Never", "Rarely", "Sometimes", "Frequently", "Always"]
)
before_bed_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Frequently": 3, "Always": 4}

st.divider()

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    raw = np.array([[
        hours_map[hours_label],
        class_map[class_label],
        entertainment_map[entertainment_label],
        dependency_map[dependency_label],
        check_noti_map[check_noti_label],
        before_bed_map[before_bed_label],
    ]], dtype=float)

    scaled = (raw - x_mean) / x_std
    X_new  = np.concatenate((np.ones((1, 1)), scaled), axis=1)
    prob   = float(sigmoid(np.matmul(X_new, theta)).item())
    pred   = int(prob >= 0.5)

    st.subheader("📊 Result")

    col1, col2 = st.columns(2)
    col1.metric("Addiction Probability", f"{prob*100:.1f}%")
    col2.metric("Prediction", "Addicted 🔴" if pred == 1 else "Not Addicted 🟢")

    if pred == 1:
        st.error(
            "⚠️ **You may be showing signs of smartphone addiction.**\n\n"
            "Consider setting screen-time limits and taking regular phone-free breaks."
        )
    else:
        st.success(
            "✅ **You do not appear to be addicted to your smartphone.**\n\n"
            "Keep maintaining a healthy balance with your device usage!"
        )

    # Probability bar
    st.progress(prob, text=f"Addiction risk: {prob*100:.1f}%")