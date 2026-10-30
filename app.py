import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import shap

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Prediction & Retention Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }
    .risk-badge {
        font-size: 1.25rem;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 8px;
        display: inline-block;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .risk-high {
        background-color: #fee2e2;
        color: #b91c1c;
        border: 1px solid #f87171;
    }
    .risk-medium {
        background-color: #fef3c7;
        color: #b45309;
        border: 1px solid #fcd34d;
    }
    .risk-low {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .stProgress > div > div > div > div {
        background-color: #4f46e5;
    }
</style>
""", unsafe_allow_html=True)


# --- Model and Asset Loading Helpers ---
def find_model_dir():
    candidates = [
        Path(__file__).resolve().parent / "models",
        Path(__file__).resolve().parent.parent / "models",
        Path.cwd() / "models",
        Path.cwd() / "Customer-Churn-Predictor" / "models"
    ]
    for c in candidates:
        if c.exists() and (c / "rf_model.pkl").exists():
            return c
    raise FileNotFoundError(f"Could not locate 'models' directory containing 'rf_model.pkl'. Checked: {candidates}")

@st.cache_resource(show_spinner="Loading trained machine learning models...")
def load_all_models():
    model_dir = find_model_dir()
    logistic_model = load(model_dir / "logistic_model.pkl")
    rf_model = load(model_dir / "rf_model.pkl")
    xgb_model = load(model_dir / "xgb_model.pkl")
    return logistic_model, rf_model, xgb_model

@st.cache_resource(show_spinner="Initializing model explainability engine...")
def load_explainer(_rf_classifier):
    return shap.TreeExplainer(_rf_classifier)

try:
    logistic_model, rf_model, xgb_model = load_all_models()
    preprocessor = rf_model.named_steps["preprocessor"]
    rf_classifier = rf_model.named_steps["model"]
    explainer = load_explainer(rf_classifier)
    models_ready = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_ready = False


# --- Caching background SHAP calculation for Tab 2 ---
@st.cache_data(show_spinner="Computing SHAP global feature importances...")
def compute_global_shap():
    # Attempt to load a representative sample from dataset if available
    data_candidates = [
        Path(__file__).resolve().parent / "data" / "Telco_Customer_Churn.csv",
        Path(__file__).resolve().parent.parent / "data" / "Telco_Customer_Churn.csv",
        Path.cwd() / "data" / "Telco_Customer_Churn.csv",
        Path.cwd() / "Customer-Churn-Predictor" / "data" / "Telco_Customer_Churn.csv"
    ]
    raw_df = None
    for dc in data_candidates:
        if dc.exists():
            try:
                raw_df = pd.read_csv(dc)
                break
            except Exception:
                pass
    
    if raw_df is not None:
        sample_df = raw_df.sample(n=min(50, len(raw_df)), random_state=42).copy()
        # Clean TotalCharges numeric column
        sample_df["TotalCharges"] = pd.to_numeric(sample_df["TotalCharges"], errors="coerce").fillna(0)
    else:
        # Fallback synthetic distribution
        sample_df = pd.DataFrame({
            "gender": np.random.choice(["Male", "Female"], 50),
            "SeniorCitizen": np.random.choice([0, 1], 50),
            "Partner": np.random.choice(["Yes", "No"], 50),
            "Dependents": np.random.choice(["Yes", "No"], 50),
            "tenure": np.random.randint(1, 72, 50),
            "PhoneService": np.random.choice(["Yes", "No"], 50),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], 50),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], 50),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], 50),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], 50),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], 50),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], 50),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], 50),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], 50),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], 50),
            "PaperlessBilling": np.random.choice(["Yes", "No"], 50),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], 50),
            "MonthlyCharges": np.random.uniform(20, 120, 50),
            "TotalCharges": np.random.uniform(100, 5000, 50)
        })

    X_sample_trans = preprocessor.transform(sample_df)
    feat_names = preprocessor.get_feature_names_out()
    X_sample_df = pd.DataFrame(X_sample_trans, columns=feat_names)
    
    # Calculate SHAP values
    shap_vals = explainer(X_sample_df)
    return X_sample_df, shap_vals, feat_names


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("Telco Churn AI")
    st.caption("End-to-End Churn Analytics & Retention Suite")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    selected_model_name = st.selectbox(
        "Active Classifier",
        ["Random Forest", "XGBoost", "Logistic Regression"],
        help="Select which ML model generates the primary prediction"
    )

    st.markdown("### 📋 Quick Profiles")
    preset = st.radio(
        "Load Preset Profile",
        ["Custom Customer", "High Risk Profile", "Loyal Long-term Customer"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("""
    **Project Highlights:**
    - 📦 Pre-trained ML Pipeline
    - 🔍 SHAP Explainable AI
    - ⚡ Fast inference via Streamlit Cache
    - 🚀 Ready for Streamlit Cloud
    """)


# --- Default values based on Preset ---
if preset == "High Risk Profile":
    d_gender, d_senior, d_partner, d_dep = "Female", 0, "No", "No"
    d_tenure = 2
    d_phone, d_multi = "Yes", "No"
    d_internet = "Fiber optic"
    d_sec, d_backup, d_prot, d_supp = "No", "No", "No", "No"
    d_tv, d_mov = "Yes", "Yes"
    d_contract, d_paperless = "Month-to-month", "Yes"
    d_payment = "Electronic check"
    d_monthly, d_total = 89.5, 179.0
elif preset == "Loyal Long-term Customer":
    d_gender, d_senior, d_partner, d_dep = "Male", 0, "Yes", "Yes"
    d_tenure = 60
    d_phone, d_multi = "Yes", "Yes"
    d_internet = "DSL"
    d_sec, d_backup, d_prot, d_supp = "Yes", "Yes", "Yes", "Yes"
    d_tv, d_mov = "No", "No"
    d_contract, d_paperless = "Two year", "No"
    d_payment = "Credit card (automatic)"
    d_monthly, d_total = 65.0, 3900.0
else:
    d_gender, d_senior, d_partner, d_dep = "Male", 0, "Yes", "No"
    d_tenure = 12
    d_phone, d_multi = "Yes", "No"
    d_internet = "DSL"
    d_sec, d_backup, d_prot, d_supp = "No", "Yes", "No", "No"
    d_tv, d_mov = "No", "No"
    d_contract, d_paperless = "Month-to-month", "Yes"
    d_payment = "Electronic check"
    d_monthly, d_total = 55.0, 660.0


# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🎯 Customer Churn Predictor", "📊 Model Insights & SHAP", "💡 Business Retention Playbook"])

# ==========================================
# TAB 1: PREDICTION
# ==========================================
with tab1:
    st.markdown("## 🔍 Customer Churn Risk Assessment")
    st.write("Fill in the customer information below to evaluate churn probability and review risk drivers.")

    with st.expander("📝 Enter Customer Profile", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if d_gender == "Male" else 1)
            senior = st.selectbox("Senior Citizen", [0, 1], index=d_senior, format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            partner = st.selectbox("Has Partner", ["Yes", "No"], index=0 if d_partner == "Yes" else 1)
            dependents = st.selectbox("Has Dependents", ["Yes", "No"], index=0 if d_dep == "Yes" else 1)
            tenure = st.slider("Tenure (Months with company)", min_value=0, max_value=72, value=d_tenure)

        with col2:
            st.markdown("#### 🌐 Subscribed Services")
            phoneservice = st.selectbox("Phone Service", ["Yes", "No"], index=0 if d_phone == "Yes" else 1)
            multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], index=["Yes", "No", "No phone service"].index(d_multi))
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(d_internet))
            onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(d_sec))
            onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(d_backup))
            deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(d_prot))
            techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(d_supp))
            streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(d_tv))
            streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], index=["Yes", "No", "No internet service"].index(d_mov))

        with col3:
            st.markdown("#### 💳 Billing & Contract")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(d_contract))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if d_paperless == "Yes" else 1)
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(d_payment)
            )
            monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=float(d_monthly), step=0.5)
            # Default auto calculation for total charges if total is roughly tenure * monthly
            auto_total = round(max(monthly, tenure * monthly), 2)
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(auto_total if d_total == 0 else d_total), step=10.0)

    # Build DataFrame
    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phoneservice],
        "MultipleLines": [multiplelines],
        "InternetService": [internet],
        "OnlineSecurity": [onlinesecurity],
        "OnlineBackup": [onlinebackup],
        "DeviceProtection": [deviceprotection],
        "TechSupport": [techsupport],
        "StreamingTV": [streamingtv],
        "StreamingMovies": [streamingmovies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    # Pick active model
    if selected_model_name == "Logistic Regression":
        active_model = logistic_model
    elif selected_model_name == "Random Forest":
        active_model = rf_model
    else:
        active_model = xgb_model

    predict_btn = st.button("🚀 Calculate Churn Probability", type="primary", use_container_width=True)

    if predict_btn and models_ready:
        prob = active_model.predict_proba(customer_data)[0][1]
        
        # Cross-model probabilities
        p_lr = logistic_model.predict_proba(customer_data)[0][1]
        p_rf = rf_model.predict_proba(customer_data)[0][1]
        p_xgb = xgb_model.predict_proba(customer_data)[0][1]

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        r_col1, r_col2 = st.columns([1, 1])

        with r_col1:
            st.metric(
                label=f"Predicted Churn Probability ({selected_model_name})",
                value=f"{prob * 100:.1f}%",
                delta=f"{(prob - 0.26) * 100:+.1f}% vs baseline (26%)",
                delta_color="inverse"
            )
            st.progress(float(prob))

            if prob > 0.60:
                st.markdown('<div class="risk-badge risk-high">⚠️ HIGH CHURN RISK</div>', unsafe_allow_html=True)
                st.error("This customer has a high likelihood of cancelling their subscription. Immediate proactive retention action is recommended.")
            elif prob > 0.30:
                st.markdown('<div class="risk-badge risk-medium">⚡ MEDIUM CHURN RISK</div>', unsafe_allow_html=True)
                st.warning("Customer exhibits moderate signs of churn. Consider sending engagement or loyalty incentives.")
            else:
                st.markdown('<div class="risk-badge risk-low">✅ LOW CHURN RISK</div>', unsafe_allow_html=True)
                st.success("Customer is stable and unlikely to churn under current conditions.")

        with r_col2:
            st.markdown("#### 🤖 Model Comparison")
            comp_df = pd.DataFrame({
                "Model": ["Random Forest", "XGBoost", "Logistic Regression"],
                "Churn Probability": [f"{p_rf*100:.1f}%", f"{p_xgb*100:.1f}%", f"{p_lr*100:.1f}%"],
                "Risk Category": [
                    "High" if p_rf > 0.6 else "Medium" if p_rf > 0.3 else "Low",
                    "High" if p_xgb > 0.6 else "Medium" if p_xgb > 0.3 else "Low",
                    "High" if p_lr > 0.6 else "Medium" if p_lr > 0.3 else "Low"
                ]
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🔍 Prediction Explanation (SHAP Waterfall)")
        st.caption("Shows how each customer attribute pushed the predicted probability higher (red) or lower (blue).")

        try:
            X_trans = preprocessor.transform(customer_data)
            feat_names = preprocessor.get_feature_names_out()
            X_trans_df = pd.DataFrame(X_trans, columns=feat_names)

            shap_values = explainer(X_trans_df)

            fig, ax = plt.subplots(figsize=(9, 4.5))
            shap.plots.waterfall(shap_values[0, :, 1], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as ex:
            st.info(f"Waterfall plot rendering notice: {ex}")


# ==========================================
# TAB 2: MODEL INSIGHTS
# ==========================================
with tab2:
    st.markdown("## 📈 Global Model Insights & Interpretability")
    st.write("Understand feature importance and global model behaviors across the entire customer dataset.")

    mi_col1, mi_col2 = st.columns(2)

    with mi_col1:
        st.subheader("🌲 Random Forest Feature Importance")
        feature_importance = rf_classifier.feature_importances_
        feat_names = preprocessor.get_feature_names_out()
        feat_imp = pd.DataFrame({
            "Feature": feat_names,
            "Importance": feature_importance
        }).sort_values("Importance", ascending=False).head(12)

        fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
        ax_fi.barh(feat_imp["Feature"][::-1], feat_imp["Importance"][::-1], color="#4f46e5")
        ax_fi.set_xlabel("Importance Score")
        ax_fi.set_title("Top 12 Drivers of Customer Churn")
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close(fig_fi)

    with mi_col2:
        st.subheader("🎯 Model Performance Comparison")
        perf_df = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "ROC AUC": [0.86, 0.85, 0.85],
            "F1 Score": [0.64, 0.65, 0.63],
            "Precision": [0.52, 0.56, 0.55],
            "Recall": [0.84, 0.78, 0.75]
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        st.info("💡 **Logistic Regression** achieves highest Recall (captures 84% of churners), while **Random Forest** provides higher Precision and balanced F1 score.")

    st.markdown("---")
    st.subheader("🐝 Global SHAP Beeswarm Plot")
    st.caption("Each dot represents a customer. Red = high feature value, Blue = low feature value. Shows impact direction on churn probability.")

    try:
        X_sample_df, sample_shap_vals, _ = compute_global_shap()
        fig_bee = plt.figure(figsize=(10, 5.5))
        shap.plots.beeswarm(sample_shap_vals[:, :, 1], max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig_bee)
        plt.close(fig_bee)
    except Exception as e:
        st.warning(f"Could not render global beeswarm plot: {e}")


# ==========================================
# TAB 3: BUSINESS RETENTION PLAYBOOK
# ==========================================
with tab3:
    st.markdown("## 💡 Data-Driven Customer Retention Playbook")

    st.markdown("""
    ### 🎯 Key Drivers Identified by Machine Learning
    
    1. **Customer Tenure** ⏳  
       *Finding:* Customers with shorter tenure (< 12 months) have the highest churn hazard. Once a customer reaches 24+ months, retention stabilizes dramatically.
       
    2. **Contract Type** 📄  
       *Finding:* Customers on **Month-to-month** contracts are over 3x more likely to churn than those on 1-year or 2-year contracts.
       
    3. **Internet Service & Tech Support** 💻  
       *Finding:* Fiber optic customers churn more frequently when they lack **Tech Support** and **Online Security** add-ons, likely due to technical friction or price sensitivity.
       
    4. **Payment Method & Billing** 💳  
       *Finding:* Customers paying via **Electronic Check** show significantly higher churn compared to automated bank transfers or credit cards.
    """)

    st.markdown("---")
    st.subheader("📋 Recommended Action Plan")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        #### 1. Onboarding Campaign
        - Deploy specialized onboarding flows during the **first 90 days**.
        - Provide dedicated customer success check-ins.
        - Offer welcome incentives for tenure milestones.
        """)
    with col_b:
        st.markdown("""
        #### 2. Contract Migration
        - Offer a **10-15% monthly discount** to transition month-to-month users into annual plans.
        - Include free security bundle trials for annual contract commitments.
        """)
    with col_c:
        st.markdown("""
        #### 3. Autopay Incentives
        - Encourage transition from electronic check to **Automated Credit Card / Bank Transfer**.
        - Offer a one-time bill credit ($5-$10) upon enabling auto-pay.
        """)
