# 📊 Customer Churn Prediction & Retention Analytics

An end-to-end Machine Learning web application built with **Streamlit**, **Scikit-Learn**, **XGBoost**, and **SHAP** to predict telecom customer churn probability and identify key drivers influencing churn.

![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-FF4B4B?style=flat&logo=Streamlit)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7-F7931E?style=flat&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-EB5424?style=flat)

---

## 🌟 Features

- **Multi-Model Churn Prediction:** Compare real-time predictions from **Random Forest**, **XGBoost**, and **Logistic Regression**.
- **SHAP Explainable AI (XAI):** Local waterfall plots explain *why* an individual customer is flagged at risk, alongside global beeswarm summary plots.
- **Preset Customer Profiles:** Instantly test High Risk, Loyal Long-term, and Custom customer scenarios.
- **Model Insights & Feature Importance:** Explore top churn drivers (tenure, contract type, payment method, tech support).
- **Business Retention Playbook:** Actionable strategies to prevent customer attrition based on data-driven findings.
- **Optimized for Streamlit Cloud:** Resource caching (`@st.cache_resource`, `@st.cache_data`) for sub-second response times.

---

## 📂 Project Structure

```text
MLProject-ChurnPrediction-main/
│
├── app.py                      # Main Streamlit Application
├── requirements.txt            # Production dependencies for deployment
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules (virtual envs, caches)
│
├── .streamlit/
│   └── config.toml             # Streamlit theme and UI configurations
│
├── models/                     # Trained ML pipelines & models
│   ├── logistic_model.pkl      # Logistic Regression pipeline
│   ├── rf_model.pkl            # Random Forest pipeline
│   └── xgb_model.pkl           # XGBoost pipeline
│
├── data/
│   └── Telco_Customer_Churn.csv # Telco customer dataset
│
└── notebooks/
    └── Customer_Churn_Prediction.ipynb # Model training & EDA notebook
```

---

## 💻 Local Setup & Execution

### 1. Clone or Navigate to the Repository
```bash
cd MLProject-ChurnPrediction-main
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows:
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
The app will automatically open in your browser at `http://localhost:8501`.

---

## ☁️ How to Deploy on Streamlit Community Cloud (Free)

Deploying takes under 3 minutes:

### Step 1: Push to GitHub
1. Make sure Git is installed on your system.
2. Initialize repository and commit:
   ```bash
   git init
   git add .
   git commit -m "Deploy: Customer Churn Prediction App"
   ```
3. Create a new repository on [GitHub](https://github.com/new) (e.g. `Customer-Churn-Predictor`).
4. Link and push your repository:
   ```bash
   git remote add origin https://github.com/<YOUR-GITHUB-USERNAME>/<YOUR-REPO-NAME>.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click **"Create app"** or **"New app"**.
3. Select your repository: `<YOUR-GITHUB-USERNAME>/<YOUR-REPO-NAME>`.
4. Set **Branch**: `main`.
5. Set **Main file path**: `app.py`.
6. Click **Deploy!** 🎉

Streamlit Cloud will automatically detect `requirements.txt`, install dependencies, and launch your live web app with a public URL!

---

## 📊 Dataset & Model Performance

The model was trained on the **Telco Customer Churn Dataset** (7,043 customer records):

| Model | ROC AUC | Recall (Capture Rate) | Precision | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.86** | **0.84** | 0.52 | 0.64 |
| **Random Forest** | 0.85 | 0.78 | **0.56** | **0.65** |
| **XGBoost** | 0.85 | 0.75 | 0.55 | 0.63 |

---

## 📜 License
This project is open-source and available under the MIT License.
