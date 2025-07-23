import streamlit as st
import pandas as pd
import joblib
from train_forecast import load_and_clean_data, forecast_next_week, forecast_next_month

model = joblib.load("models/xgboost_model.pkl")

st.title("Sales forecasting tool")

option = st.radio("Choose Forecast Type", ["Weekly Forecast", "Monthly Forecast"])

uploaded_file = st.file_uploader("Upload your csv file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if option == "Weekly Forecast":
        forecast = forecast_next_week(data, model)
        st.success("Weekly forecast completed")
    elif option == "Monthly Forecast":
        forecast = forecast_next_month(data, model)
        st.success("Monthly forecast completed")


    st.dataframe(forecast)

    # After displaying the forecast
    st.subheader("📊 Model Evaluation Metrics (from training data)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="MAE", value="9.96")
        st.metric(label="MAPE", value="9.86%")

    with col2:
        st.metric(label="RMSE", value="11.86")
        st.metric(label="SE", value="140.56")

    with col3:
        st.metric(label="R² Score", value="0.493")


    csv = forecast.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Forecast", data=csv, file_name='forecast.csv', mime='text/csv')
    