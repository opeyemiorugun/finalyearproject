import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_model():
    model = joblib.load("model/theft_detection_model.pkl")
    return model

def app():
    st.title("Electricity Theft Detection")

    st.write("Upload the whole house power data.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state['whole_house_data'] = data
        st.write("Data uploaded successfully!")
        st.write(data.head())
    else:
        st.write("No data uploaded yet.")
        return

    model = load_model()

    def pre_process(data):
        #Input the code for the preprocessing of the data
        return data
    
    # Example function for theft detection (replace with your actual code)
    def detect_theft(data, model):
        # Dummy predictions (replace with actual model predictions)
        predictions = model.predict(data)
        return predictions

    predictions = detect_theft(data, model)

    # Calculate and display metrics
    actuals = data['actual_theft']
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)

    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")

    st.write("Theft Detection Results")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['power_usage'], label="Power Usage")
    ax.plot(data.index, predictions, label="Theft Prediction")
    ax.legend()
    st.pyplot(fig)

    # Example historical comparison (replace with your actual comparison code)
    st.write("Historical Comparison")
    st.write("Energy consumption is 30% higher than the same period last year.")
