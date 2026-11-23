from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/fraud_model.joblib")
SCALER_PATH = Path("models/scaler.joblib")
# Optional: if you saved two separate scalers in the notebook, put them here
AMOUNT_SCALER_PATH = Path("models/amount_scaler.joblib")
TIME_SCALER_PATH = Path("models/time_scaler.joblib")

st.set_page_config(
    page_title="Credit Card Fraud Detector - Machine Learning & Risk Scoring",
    page_icon="💳",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sample scenarios
# ---------------------------------------------------------------------------
NORMAL_SAMPLE = {
    "amount_val": 25.80, "time_val": 45000.0,
    "v1_val": -0.5, "v4_val": 0.2, "v10_val": 0.1,
    "v11_val": -0.3, "v12_val": 0.4, "v14_val": 0.2, "v17_val": 0.1,
}
FRAUD_SAMPLE = {
    "amount_val": 529.00, "time_val": 406.0,
    "v1_val": -2.31, "v4_val": 4.00, "v10_val": -2.77,
    "v11_val": 3.20, "v12_val": -2.90, "v14_val": -4.29, "v17_val": -2.83,
}
DEFAULTS = {
    "amount_val": 45.50, "time_val": 85000.0,
    "v1_val": 0.0, "v4_val": 0.0, "v10_val": 0.0,
    "v11_val": 0.0, "v12_val": 0.0, "v14_val": 0.0, "v17_val": 0.0,
}


def apply_sample(sample: dict):
    """Button callback: writes straight into the widget keys."""
    for key, value in sample.items():
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_fraud_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    amount_scaler = joblib.load(AMOUNT_SCALER_PATH) if AMOUNT_SCALER_PATH.exists() else None
    time_scaler = joblib.load(TIME_SCALER_PATH) if TIME_SCALER_PATH.exists() else None
    return model, scaler, amount_scaler, time_scaler


def get_feature_order(model):
    """Return the exact column order the model was trained on (if available)."""
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    try:  # XGBoost
        names = model.get_booster().feature_names
        if names:
            return list(names)
    except Exception:
        pass
    try:  # LightGBM
        names = model.booster_.feature_name()
        if names:
            return list(names)
    except Exception:
        pass
    # Fallback: common order used in this project
    return [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]


def scale_amount_and_time(amount, time_sec, scaler, amount_scaler, time_scaler):
    """Scale Amount and Time correctly depending on how the scaler was saved."""
    # Case 1: two dedicated scalers were saved
    if amount_scaler is not None and time_scaler is not None:
        return (
            float(amount_scaler.transform([[amount]])[0][0]),
            float(time_scaler.transform([[time_sec]])[0][0]),
        )

    n_features = getattr(scaler, "n_features_in_", 1)

    # Case 2: one scaler fitted on two columns
    if n_features == 2:
        names = list(getattr(scaler, "feature_names_in_", ["Amount", "Time"]))
        values = {"Amount": amount, "Time": time_sec}
        row = [values.get(n, v) for n, v in zip(names, [amount, time_sec])]
        out = scaler.transform([row])[0]
        scaled = dict(zip(names, out))
        return float(scaled.get("Amount", out[0])), float(scaled.get("Time", out[1]))

    # Case 3: one scaler fitted on a single column (original behaviour)
    return (
        float(scaler.transform([[amount]])[0][0]),
        float(scaler.transform([[time_sec]])[0][0]),
    )


def main():
    st.title("💳 Credit Card Fraud Detector - Machine Learning & Risk Scoring")
    st.caption("Cost-Sensitive Imbalanced Classification & Dynamic Thresholding Engine.")

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error("Model assets not found! Please run the notebook first so that "
                 f"`{MODEL_PATH}` and `{SCALER_PATH}` exist.")
        st.stop()

    model, scaler, amount_scaler, time_scaler = load_fraud_assets()

    # Initialise widget state once
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    with st.sidebar:
        st.header("Risk Threshold Control")
        decision_threshold = st.slider(
            "Fraud Decision Threshold",
            min_value=0.05, max_value=0.95, value=0.30, step=0.05,
            help="Lower threshold boosts RECALL (catches more fraud but increases "
                 "false alerts). Higher threshold boosts PRECISION.",
        )

    st.subheader("Try Sample Transaction Scenarios")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.button("Sample Normal Transaction", use_container_width=True,
                  on_click=apply_sample, args=(NORMAL_SAMPLE,))
    with btn_col2:
        st.button("Sample Suspected Fraud", use_container_width=True,
                  on_click=apply_sample, args=(FRAUD_SAMPLE,))

    st.subheader("Key Transaction Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=10.0, key="amount_val")
        time_sec = st.number_input("Time Elapsed (Seconds)", min_value=0.0, step=1000.0, key="time_val")
        v1 = st.number_input("PCA Feature V1", step=0.1, key="v1_val")
        v4 = st.number_input("PCA Feature V4", step=0.1, key="v4_val")

    with col2:
        v10 = st.number_input("PCA Feature V10 (Key Fraud Indicator)", step=0.1, key="v10_val")
        v11 = st.number_input("PCA Feature V11 (Key Fraud Indicator)", step=0.1, key="v11_val")
        v12 = st.number_input("PCA Feature V12 (Key Fraud Indicator)", step=0.1, key="v12_val")

    with col3:
        v14 = st.number_input("PCA Feature V14 (Top Fraud Feature)", step=0.1, key="v14_val")
        v17 = st.number_input("PCA Feature V17 (Top Fraud Feature)", step=0.1, key="v17_val")

    st.caption("Remaining PCA features (V2, V3, V5-V9, V13, V15, V16, V18-V28) are set to 0 "
               "for this demo.")
    st.divider()

    if st.button("Analyze Transaction Fraud Risk", use_container_width=True, type="primary"):
        try:
            scaled_amount, scaled_time = scale_amount_and_time(
                amount, time_sec, scaler, amount_scaler, time_scaler
            )

            feature_dict = {f"V{i}": 0.0 for i in range(1, 29)}
            feature_dict.update({
                "V1": v1, "V4": v4, "V10": v10, "V11": v11,
                "V12": v12, "V14": v14, "V17": v17,
                "scaled_amount": scaled_amount,
                "scaled_time": scaled_time,
            })

            # Force the exact column order used during training
            feature_order = get_feature_order(model)
            input_df = pd.DataFrame([feature_dict]).reindex(columns=feature_order, fill_value=0.0)

            fraud_prob = float(model.predict_proba(input_df)[0][1])
            fraud_prob = min(max(fraud_prob, 0.0), 1.0)
            is_fraud = fraud_prob >= decision_threshold
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.subheader("Transaction Risk Assessment")

        if is_fraud:
            st.error(f"🚨 **FRAUDULENT TRANSACTION FLAGGED** — Fraud Risk Score: {fraud_prob * 100:.1f}%")
        else:
            st.success(f"✅ **TRANSACTION APPROVED (Legitimate)** — Fraud Risk Score: {fraud_prob * 100:.1f}%")

        st.progress(
            fraud_prob,
            text=f"Calculated Fraud Probability: {fraud_prob * 100:.1f}% "
                 f"(Threshold: {decision_threshold * 100:.0f}%)",
        )


if __name__ == "__main__":
    main()
