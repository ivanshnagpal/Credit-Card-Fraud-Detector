from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/fraud_model.joblib")
SCALER_PATH = Path("models/scaler.joblib")

st.set_page_config(
    page_title="💳 Credit Card Fraud Detector AI",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_fraud_assets():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error("Model assets not found! Please run the notebook first.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def main():
    st.title("💳 Credit Card Fraud Detector AI")
    st.caption("Cost-Sensitive Imbalanced Classification & Dynamic Thresholding Engine.")

    model, scaler = load_fraud_assets()

    with st.sidebar:
        st.header("⚙️ Risk Threshold Control")
        decision_threshold = st.slider(
            "Fraud Decision Threshold", 
            min_value=0.05, max_value=0.95, value=0.30, step=0.05,
            help="Lower threshold boosts RECALL (catches more fraud but increases false alerts). Higher threshold boosts PRECISION."
        )

    st.subheader("💡 Try Sample Transaction Scenarios")
    btn_col1, btn_col2 = st.columns(2)
    
    if "amount_val" not in st.session_state:
        st.session_state.amount_val = 45.50
    if "time_val" not in st.session_state:
        st.session_state.time_val = 85000.0
    if "v1_val" not in st.session_state:
        st.session_state.v1_val = 0.0
    if "v4_val" not in st.session_state:
        st.session_state.v4_val = 0.0
    if "v10_val" not in st.session_state:
        st.session_state.v10_val = 0.0
    if "v11_val" not in st.session_state:
        st.session_state.v11_val = 0.0
    if "v12_val" not in st.session_state:
        st.session_state.v12_val = 0.0
    if "v14_val" not in st.session_state:
        st.session_state.v14_val = 0.0
    if "v17_val" not in st.session_state:
        st.session_state.v17_val = 0.0

    with btn_col1:
        if st.button("✅ Sample Normal Transaction", use_container_width=True):
            st.session_state.amount_val = 25.80
            st.session_state.time_val = 45000.0
            st.session_state.v1_val = -0.5
            st.session_state.v4_val = 0.2
            st.session_state.v10_val = 0.1
            st.session_state.v11_val = -0.3
            st.session_state.v12_val = 0.4
            st.session_state.v14_val = 0.2
            st.session_state.v17_val = 0.1

    with btn_col2:
        if st.button("🚨 Sample Suspected Fraud", use_container_width=True):
            # Real Fraud pattern from Credit Card dataset
            st.session_state.amount_val = 529.00
            st.session_state.time_val = 406.00
            st.session_state.v1_val = -2.31
            st.session_state.v4_val = 4.00
            st.session_state.v10_val = -2.77
            st.session_state.v11_val = 3.20
            st.session_state.v12_val = -2.90
            st.session_state.v14_val = -4.29
            st.session_state.v17_val = -2.83

    st.subheader("📋 Key Transaction Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount = st.number_input("Transaction Amount ($)", value=float(st.session_state.amount_val), min_value=0.0, step=10.0)
        time_sec = st.number_input("Time Elapsed (Seconds)", value=float(st.session_state.time_val), min_value=0.0, step=1000.0)
        v1 = st.number_input("PCA Feature V1", value=float(st.session_state.v1_val), step=0.1)
        v4 = st.number_input("PCA Feature V4", value=float(st.session_state.v4_val), step=0.1)

    with col2:
        v10 = st.number_input("PCA Feature V10 (Key Fraud Indicator)", value=float(st.session_state.v10_val), step=0.1)
        v11 = st.number_input("PCA Feature V11 (Key Fraud Indicator)", value=float(st.session_state.v11_val), step=0.1)
        v12 = st.number_input("PCA Feature V12 (Key Fraud Indicator)", value=float(st.session_state.v12_val), step=0.1)

    with col3:
        v14 = st.number_input("PCA Feature V14 (Top Fraud Feature)", value=float(st.session_state.v14_val), step=0.1)
        v17 = st.number_input("PCA Feature V17 (Top Fraud Feature)", value=float(st.session_state.v17_val), step=0.1)

    st.divider()

    if st.button("🔍 Analyze Transaction Fraud Risk", use_container_width=True):
        scaled_amount = scaler.transform([[amount]])[0][0]
        scaled_time = scaler.transform([[time_sec]])[0][0]
        
        feature_dict = {f"V{i}": 0.0 for i in range(1, 29)}
        feature_dict["V1"] = v1
        feature_dict["V4"] = v4
        feature_dict["V10"] = v10
        feature_dict["V11"] = v11
        feature_dict["V12"] = v12
        feature_dict["V14"] = v14
        feature_dict["V17"] = v17
        feature_dict["scaled_amount"] = scaled_amount
        feature_dict["scaled_time"] = scaled_time
        
        input_df = pd.DataFrame([feature_dict])
        
        fraud_prob = float(model.predict_proba(input_df)[0][1])
        is_fraud = fraud_prob >= decision_threshold

        st.subheader("📊 Transaction Risk Assessment")
        
        if is_fraud:
            st.error(f"🚨 **FRAUDULENT TRANSACTION FLAGGED** — Fraud Risk Score: {fraud_prob * 100:.1f}%")
        else:
            st.success(f"✅ **TRANSACTION APPROVED (Legitimate)** — Fraud Risk Score: {fraud_prob * 100:.1f}%")

        st.progress(fraud_prob, text=f"Calculated Fraud Probability: {fraud_prob * 100:.1f}% (Threshold: {decision_threshold * 100:.0f}%)")

if __name__ == "__main__":
    main()