import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ========== Load Model ==========
model = joblib.load("model/xgboost_fraud_model.pkl")

# ========== Feature Definitions ==========
features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
            'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
            'V28', 'Amount']

inputs_config = {
    "Transaction Velocity (V14)": ("V14", {"Low": -5.0, "Medium": 0.0, "High": 5.0}),
    "Amount Deviation (V12)": ("V12", {"Low": -10.0, "Medium": 0.0, "High": 10.0}),
    "Transaction Frequency (V10)": ("V10", {"Rare": -5.0, "Normal": 0.0, "Frequent": 5.0}),
    "Account Age (V17)": ("V17", {"New": -5.0, "Established": 0.0, "Old": 5.0}),
    "Transaction Amount (₹)": ("Amount", None)
}

# ========== Page Config ==========
st.set_page_config("CredGuard", page_icon="💳", layout="centered")

# ========== Sidebar Navigation ==========
with st.sidebar:
    st.title("💳 CredGuard")
    page = st.radio("Navigation", ["Predict", "Risk", "Report", "About"])
    country = st.selectbox("🌍 Country", ["India", "USA", "UK", "Germany", "Others"])
    device = st.selectbox("💻 Device", ["Mobile", "Desktop", "ATM"])
    time = st.selectbox("⏰ Time", ["Morning", "Afternoon", "Evening", "Night"])

# ========== Input Form ==========
def get_inputs():
    data = {}
    for label, (col, options) in inputs_config.items():
        if options:
            choice = st.selectbox(label, list(options.keys()))
            data[col] = options[choice]
        else:
            data[col] = st.number_input(label, 0.0, 100000.0, 100.0, 10.0)
    return data

# ========== Predict Page ==========
if page == "Predict":
    st.header("🚀 Predict Transaction")
    user_input = get_inputs()

    # Build model input
    input_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    for k, v in user_input.items():
        input_df.at[0, k] = v

    st.markdown("### 📋 Transaction Summary")
    st.dataframe(input_df)

    if st.button("🔎 Run Prediction"):
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1]

        # Save session data
        st.session_state.prediction = prediction[0]
        st.session_state.proba = proba
        st.session_state.user_input = user_input
        st.session_state.meta = {
            "Country": country,
            "Device": device,
            "Time": time
        }

        # Output
        if prediction[0]:
            st.error("⚠️ Fraud Detected")
        else:
            st.success("✅ Legit Transaction")

# ========== Risk Page ==========
elif page == "Risk":
    st.header("📊 Risk Analysis")
    if "prediction" in st.session_state:
        proba = st.session_state.proba
        st.metric("Fraud Score", f"{proba:.2%}")

        if proba > 0.9:
            st.error("🚨 Very High Risk! Likely fraudulent transaction.")
        elif proba > 0.6:
            st.warning("⚠️ Moderate to High Risk — needs review.")
        elif proba > 0.3:
            st.info("🔍 Slightly suspicious. Monitor closely.")
        else:
            st.success("✅ No major risk detected.")

        with st.expander("📄 Metadata"):
            for k, v in st.session_state.meta.items():
                st.write(f"**{k}**: {v}")
    else:
        st.info("Make a prediction first.")

# ========== Report Page ==========
elif page == "Report":
    st.header("📄 Transaction Report")
    if "user_input" in st.session_state:
        u = st.session_state.user_input
        meta = st.session_state.meta
        pred = "Fraud" if st.session_state.prediction else "Legit"

        st.write(f"""
        **Country:** {meta['Country']}  
        **Device:** {meta['Device']}  
        **Time of Day:** {meta['Time']}  
        **Amount:** ₹{u.get('Amount', 'N/A')}  
        **Prediction:** {pred}
        """)

        report_text = f"""
CredGuard Fraud Detection Report

Country: {meta['Country']}
Device: {meta['Device']}
Time of Day: {meta['Time']}
Amount: ₹{u.get('Amount', 'N/A')}
Prediction: {pred}
Fraud Probability: {st.session_state.proba:.2%}
"""

        st.download_button("📥 Download Report", report_text, file_name="transaction_report.txt")
    else:
        st.info("No data available.")

# ========== About Page ==========
elif page == "About":
    st.header("👤 About")
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.markdown("""
Welcome to **CredGuard** — a real-time credit card fraud detection tool built using Machine Learning.

### 🚀 What does this app do?
- Predicts whether a transaction is legitimate or fraudulent  
- Gives a fraud probability score and insight  
- Allows report generation and export  

### 🧠 Tech Stack:
- **Streamlit** for frontend UI  
- **XGBoost** as the fraud detection model  
- **Python, pandas, joblib** for backend logic  

### 👨‍💻 Developed by:
**Mohit Soni**  
Machine Learning & Cybersecurity Enthusiast  

> "Security is not just a feature. It’s the foundation."

---
Want to collaborate or hire? [Connect on LinkedIn](https://www.linkedin.com/in/mohit-soni-819223316/)
""")
