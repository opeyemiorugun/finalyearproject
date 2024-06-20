import streamlit as st
from multiapp import MultiApp
from apps import home, power_forecasting, electricity_theft_detection, energy_optimization

# Set page config at the top of the main script
st.set_page_config(
    page_title="Appliance Load Data Analysis",
    layout="wide"
    page_icon="💡",
)

app = MultiApp()

# Add all application modules here
app.add_app("Home", home.app)
app.add_app("Power Forecasting", power_forecasting.app)
app.add_app("Electricity Theft Detection", electricity_theft_detection.app)
app.add_app("Energy Optimization", energy_optimization.app)

# The main app
app.run()
