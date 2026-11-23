# Credit Card Fraud Detector - Machine Learning & Risk Scoring 

An End-to-End machine learning application for identifying fraudulent credit card transactions in highly imbalanced data (0.17% fraud rate). Built using **Scikit-Learn**, **XGBoost**, **Random Forest**, **RobustScaler**, **KaggleHub**, and deployed via an interactive **Streamlit** web application.

---

## Project Overview

Credit card fraud detection is a classic **high-imbalance classification** problem. Out of 284,807 transactions in the dataset, only 492 (0.17%) are fraudulent. Predicting every transaction as "Legitimate" yields a misleading **99.83% accuracy** while missing 100% of actual fraud cases.

This project uses **Cost-Sensitive Learning (Class Weights)** to penalize minority fraud errors, evaluates models using **Precision-Recall Area Under Curve (PR-AUC)** and **ROC-AUC**, and deploys a Streamlit application featuring a dynamic **Decision Threshold Slider** to balance Precision vs. Recall.

---

## Key Features

- **Cost-Sensitive Class Weighting:** Trains Random Forest with `class_weight='balanced_subsample'` and XGBoost with `scale_pos_weight` to address extreme 99.83% to 0.17% imbalance.
- **Outlier-Robust Scaling:** Standardizes skewed `Amount` and `Time` features using `RobustScaler` (median & IQR).
- **Precision-Recall AUC Focus:** Evaluates model performance using PR-AUC and ROC-AUC curves rather than raw accuracy.
- **Dynamic Threshold Slider:** Sidebar control allowing users to adjust the decision threshold ($0.05$ to $0.95$) to prioritize **Recall** (catching more fraud) or **Precision** (minimizing false alarms).
- **Instant Test Presets:** Preset buttons for **Sample Normal Transaction** and **Sample Suspected Fraud** (derived from authentic fraud PCA feature vectors: `V10`, `V11`, `V12`, `V14`, `V17`).

---

## Dataset Information

The model is trained on the benchmark **Credit Card Fraud Detection Dataset** (Kaggle / ULB).

- **Kaggle Link:** [Credit Card Fraud Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Records:** 284,807 transactions (September 2013)
- **Class Skewness:**
  - `0` (Legitimate): 284,315 transactions (99.83%)
  - `1` (Fraudulent): 492 transactions (0.17%)
- **Attributes:**
  - `Time`: Seconds elapsed between this transaction and the first transaction in dataset.
  - `Amount`: Transaction monetary amount ($).
  - `V1` to `V28`: PCA-transformed numerical feature components (anonymized for privacy).
  - `Class`: Target binary label (`1` = Fraud, `0` = Legitimate).

---

## Project Structure

```text
Credit Card Fraud/
│
├── data/
│   └── creditcard.csv             # Kaggle dataset file (uncompressed ~150 MB)
│
├── models/
│   ├── fraud_model.joblib         # Saved tuned Random Forest / XGBoost model
│   └── scaler.joblib              # Saved RobustScaler pipeline
│
├── notebooks/
│   └── fraud_detection.ipynb      # Notebook (EDA, Class Weights, PR-AUC, ROC-AUC, Thresholding)
│
├── app.py                         # Interactive Streamlit Web Application
├── requirements.txt               # Dependencies list
├── .gitignore                     # Git ignore file
└── README.md                      # Complete project documentation
```

---

## Installation & Local Setup

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Navigate to Project Directory
```bash
cd "d:\Internship\HawkLogix\DL\Credit Card Fraud"
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Train Model & Save Assets
Open and execute all cells in `notebooks/fraud_detection.ipynb` to generate `models/fraud_model.joblib` and `models/scaler.joblib`.

> **Note:** The notebook uses `kagglehub` to automatically fetch `creditcard.csv` from Kaggle.

### 5. Launch Streamlit App
```bash
streamlit run app.py
```
The application will open automatically at `http://localhost:8501`.

---

## How to Use the Web App

1. Adjust the **Fraud Decision Threshold** slider on the sidebar ($0.05$ to $0.95$). Lowering the threshold boosts **Recall** (catches more fraud).
2. Click **Sample Normal Transaction** or **Sample Suspected Fraud** for instant pre-filled feature parameters.
3. Or enter custom transaction parameters ($Amount$, $Time$, $V1$, $V4$, $V10$, $V11$, $V12$, $V14$, $V17$).
4. Click **Analyze Transaction Fraud Risk**.
5. View the risk assessment (**FRAUDULENT TRANSACTION FLAGGED** vs **TRANSACTION APPROVED**) and the fraud probability progress meter.

---

## Tech Stack

- **Frontend / Deployment:** Streamlit
- **Machine Learning Frameworks:** Scikit-Learn, XGBoost, Imbalanced-Learn
- **Data Acquisition & Preprocessing:** KaggleHub, Joblib, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## Connect

**Vansh Nagpal**

[LinkedIn](https://www.linkedin.com/in/vansh-nagpal-vn011) • [GitHub](https://github.com/ivanshnagpal)
