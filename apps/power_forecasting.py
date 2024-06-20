import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import lightgbm as lgm
from sklearn.ensemble import HistGradientBoostingRegressor
from utils import load_model, preprocess_data, forecast_power, calculate_metrics, plot_forecast_vs_actual, plot_feature_importance

def app():
    st.title("Power Forecasting")

    if 'uploaded_data' not in st.session_state:
        st.write("Please upload data on the Home page first.")
        return

    data = st.session_state['uploaded_data']
    column_names = st.session_state['column_names']
    st.write("Data loaded successfully!")

    # Preprocess the uploaded data
    preprocessed_data = preprocess_data(data, column_names)
    
    # Allow users to select appliances for forecasting
    selected_appliances = st.multiselect(
        'Select appliances to forecast', column_names, default=column_names[:1]
    )

    if not selected_appliances:
        st.write("Please select at least one appliance to forecast.")
        return

    model = load_model(selected_appliances)

    horizon_options = ["Next day", "Next week", "Next month"]
    horizon = st.selectbox("Select forecasting horizon", horizon_options)

    horizon_mapping = {"Next day": 1, "Next week": 7, "Next month": 30}
    forecast_horizon = horizon_mapping[horizon]
    
    with st.spinner("Generating forecasts..."):
        predictions = forecast_power(preprocessed_data, selected_appliances, model, forecast_horizon)

    actuals = preprocessed_data.iloc[-forecast_horizon:]

    st.write("Forecast vs Actual")
    plot_forecast_vs_actual(actuals, predictions, selected_appliances)

    mae, rmse = calculate_metrics(actuals, predictions)
    st.write(f"MAE: {mae}")
    st.write(f"RMSE: {rmse}")

    if hasattr(model, "feature_importances_"):
        st.write("Feature Importance")
        plot_feature_importance(model.feature_importances_, preprocessed_data.columns)
    else:
        st.write("Model does not support feature importance.")

    st.session_state['forecasted_data'] = predictions


# Main entry point for the app
if __name__ == "__main__":
    app()
