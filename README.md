# credit-card-fraud
# 💳 CredGuard - Credit Card Fraud Detection

CredGuard is a real-time credit card fraud detection system powered by **XGBoost** and built using **Streamlit**. This interactive web app predicts whether a transaction is **fraudulent or legitimate** based on user-provided features.

---

## 🚀 Features

- 🔍 Predict transaction legitimacy using a trained XGBoost model
- 📊 View risk levels with visual fraud probability
- 📄 Generate and download transaction reports
- 🧠 Real-time user input mapping to model features
- 🎨 Clean and intuitive UI with custom dropdowns and metadata
- 🌐 Works entirely locally or deployable on Streamlit Cloud

---

## 🧠 Tech Stack

- **Streamlit** – Frontend Web Interface
- **XGBoost** – Fraud Detection Model
- **Pandas / NumPy** – Data Handling
- **Joblib** – Model Loading
- **Python 3.8+**

---

## 📦 Folder Structure

CredGuard_Project/
│
├── main.py # Streamlit app
├── model/
│ └── xgboost_fraud_model.pkl # Pretrained model file
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── Screenshot.png # Optional: UI preview


---

## 🧪 How to Run

1. **Clone the repository** or download the folder:

   ```bash
   git clone https://github.com/your-username/CredGuard.git
   cd CredGuard

   pip install -r requirements.txt
streamlit run main.py
