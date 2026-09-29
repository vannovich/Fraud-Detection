import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

model = load_model()

st.title("Fraud Detection Prediction App")
st.markdown("Please enter the transaction details and use the predict button to get the prediction.")

st.divider()

transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=10000.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=9000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }])

    prediction = int(model.predict(input_data)[0])

    st.subheader(f"Prediction: {prediction}")

    if hasattr(model, "predict_proba"):
        fraud_prob = model.predict_proba(input_data)[0][1]
        st.write(f"Fraud probability: {fraud_prob:.2%}")

    if prediction == 1:
        st.error("The transaction is fraudulent.")
    else:
        st.success("The transaction is not fraudulent.")