import streamlit as st
import joblib

model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction")

tenure = st.number_input("Tenure (months)")
monthly_charges = st.number_input("Monthly Charges")

if st.button("Predict"):
    prediction = model.predict([[tenure, monthly_charges]])
    
    if prediction[0] == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")
