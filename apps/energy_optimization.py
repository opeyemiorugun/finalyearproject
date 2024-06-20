import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utils import minimize_standby, load_balancing, lifetime_extension, minimize_energy_cost, minimize_noise, optimize_energy_usage
# Include all the optimization functions and the main optimization function

def app():
    st.title("Energy Optimization")

    if 'forecasted_data' not in st.session_state:
        st.write("Please run the power forecasting page first.")
        return

    forecasted_data = st.session_state['forecasted_data']
    column_names = st.session_state['column_names']

    st.write("Optimization Goals")
    goals = st.multiselect("Select Optimization Goals", [
        "Minimize Standby Power Consumption", 
        "Load Balancing", 
        "Minimize Noise Disruption", 
        "Appliance Lifetime Extension",
        "Minimize Energy Costs"
    ])

    st.write("Select Appliances for Optimization")
    appliances = forecasted_data.columns.tolist()
    selected_appliances = st.multiselect("Select appliances to optimize", appliances, default=appliances)

    usage_guidelines = {
        'washer_dryer': {'max_daily_usage': 2, 'cool_down_period': timedelta(hours=1)},
        'oven': {'max_daily_usage': 3, 'cool_down_period': timedelta(hours=2)},
        # Add guidelines for other appliances as needed
    }

    faux_time_of_use_rates = {
        'peak': 0.30,
        'off_peak': 0.10,
        'super_off_peak': 0.05
    }

    rate_schedule = {
        'peak': [(17, 21)],  # 5 PM to 9 PM
        'off_peak': [(7, 17), (21, 23)],  # 7 AM to 5 PM, 9 PM to 11 PM
        'super_off_peak': [(0, 7)]  # 12 AM to 7 AM
    }

    noise_categories = {
        'stereo_speakers_bedroom': 'noisy',
        'i7_desktop': 'moderate',
        'hairdryer': 'noisy',
        'primary_tv': 'moderate',
        '24_inch_lcd_bedroom': 'quiet',
        'treadmill': 'noisy',
        'network_attached_storage': 'quiet',
        'core2_server': 'quiet',
        '24_inch_lcd': 'quiet',
        'PS4': 'moderate',
        'steam_iron': 'noisy',
        'nespresso_pixie': 'moderate',
        'atom_pc': 'quiet',
        'toaster': 'moderate',
        'home_theatre_amp': 'noisy',
        'sky_hd_box': 'quiet',
        'kettle': 'moderate',
        'fridge_freezer': 'quiet',
        'oven': 'moderate',
        'electric_hob': 'moderate',
        'dishwasher': 'noisy',
        'microwave': 'moderate',
        'washer_dryer': 'noisy',
        'vacuum_cleaner': 'noisy'
    }

    if st.button("Run Optimization"):
        optimized_data, changes_logs = optimize_energy_usage(
            forecasted_data, goals, selected_appliances, 
            usage_guidelines, faux_time_of_use_rates, 
            rate_schedule, noise_categories
        )

        st.write("Optimized Results")
        for appliance in selected_appliances:
            st.write(f"Appliance: {appliance}")
            fig, ax = plt.subplots()
            ax.plot(forecasted_data.index, forecasted_data[appliance], label="Forecasted")
            ax.plot(optimized_data.index, optimized_data[appliance], label="Optimized")
            ax.legend()
            st.pyplot(fig)

        st.session_state['optimized_data'] = optimized_data

        st.write("Optimization Changes Logs")
        for goal, log in changes_logs:
            st.write(f"Goal: {goal}")
            log_df = pd.DataFrame(log, columns=['Appliance', 'Time_Changed'])
            st.write(log_df)
