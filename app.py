import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_breast_cancer

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model + Dataset
# -----------------------------
model = joblib.load("breast_cancer_model.pkl")

data = load_breast_cancer()

features = list(data.feature_names)

default_values = pd.DataFrame(
    data.data,
    columns=features
).median()

# -----------------------------
# SHAP Explainer
# -----------------------------
@st.cache_resource
def create_shap_explainer():

    scaler = model.named_steps["scaler"]
    logistic_model = model.named_steps["model"]

    X_scaled = scaler.transform(data.data)

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=features
    )

    masker = shap.maskers.Independent(
        X_scaled_df,
        max_samples=100
    )

    explainer = shap.LinearExplainer(
        logistic_model,
        masker
    )

    return explainer


explainer = create_shap_explainer()

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Breast Cancer Prediction")

st.write(
    "Machine Learning based breast cancer classification "
    "using Logistic Regression."
)

st.warning(
    "⚠️ Educational project only. "
    "This is NOT a medical diagnostic tool."
)

st.divider()

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    st.subheader("🔢 Tumor Measurements")

    st.caption(
        "Default values are dataset median values. "
        "Modify them and click Predict."
    )

    inputs = {}

    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(features):

        value = float(default_values[feature])

        if i % 3 == 0:

            with col1:
                inputs[feature] = st.number_input(
                    feature,
                    value=value,
                    format="%.6f"
                )

        elif i % 3 == 1:

            with col2:
                inputs[feature] = st.number_input(
                    feature,
                    value=value,
                    format="%.6f"
                )

        else:

            with col3:
                inputs[feature] = st.number_input(
                    feature,
                    value=value,
                    format="%.6f"
                )

    submitted = st.form_submit_button(
        "🔍 Predict",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:

    input_data = pd.DataFrame(
        [inputs],
        columns=features
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    malignant_probability = probabilities[0]
    benign_probability = probabilities[1]

    # -------------------------
    # Prediction Result
    # -------------------------
    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction == 0:

        st.error("🔴 Prediction: Malignant")

    else:

        st.success("🟢 Prediction: Benign")

    # -------------------------
    # Probabilities
    # -------------------------
    st.write("### Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Malignant Probability",
            f"{malignant_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Benign Probability",
            f"{benign_probability * 100:.2f}%"
        )

    st.progress(float(benign_probability))

    st.caption(
        "These are model-estimated probabilities, "
        "not medical certainty."
    )

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    st.divider()

    st.subheader("🧠 Why did the model make this prediction?")

    st.write(
        "SHAP explains how individual features influenced "
        "the model's prediction for this particular sample."
    )

    # Scale input using the same scaler used during training
    scaler = model.named_steps["scaler"]

    input_scaled = scaler.transform(input_data)

    input_scaled_df = pd.DataFrame(
        input_scaled,
        columns=features
    )

    # Calculate SHAP values
    shap_values = explainer(input_scaled_df)

    # Waterfall plot
    fig = plt.figure(figsize=(10, 7))

    shap.plots.waterfall(
        shap_values[0],
        max_display=10,
        show=False
    )

    st.pyplot(fig, clear_figure=True)

    st.info(
        "💡 SHAP values show how each feature influenced "
        "the model's prediction. They indicate model behavior, "
        "not medical causation."
    )