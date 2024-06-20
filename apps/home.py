import streamlit as st
import pandas as pd
import requests
from io import StringIO

def fetch_github_file(url):
    st.write(f"Fetching URL: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        st.write(f"Successfully fetched file from {url}")
        return StringIO(response.text), url.split('/')[-1]  # Return file-like object and filename
    else:
        st.error(f"Failed to fetch file from {url} with status code {response.status_code}")
        return None, None

def load_data(uploaded_files):
    label_file = uploaded_files['label_file']
    labels = pd.read_csv(label_file, delimiter="\s+", names=["no", "title"])
    label_dict = labels.set_index('no').to_dict()['title']

    dataframes_list = []
    column_names = []

    for uploaded_file, filename in uploaded_files['csv_files']:
        appliance_number = int(filename.split('_')[1].split('.')[0])

        # Debug: Show appliance number and label dictionary
        # st.write(f"Appliance number from file: {appliance_number}")
        # st.write(f"Label dictionary: {label_dict}")

        # Check if appliance_number exists in label_dict
        if appliance_number not in label_dict:
            st.error(f"Appliance number {appliance_number} not found in label file.")
            continue

        appliance_name = label_dict[appliance_number]
        column_names.append(appliance_name)

        temp = pd.read_csv(uploaded_file, delimiter="\s+", names=["timestamp", "Power"], dtype={'timestamp': 'float64'}, engine='python')
        temp['datetime'] = pd.to_datetime(temp['timestamp'], unit='s')
        temp.drop(columns=['timestamp'], inplace=True)
        temp.set_index('datetime', inplace=True)
        temp.columns = [appliance_name]
        dataframes_list.append(temp)

    df = pd.concat(dataframes_list, axis=1) if dataframes_list else pd.DataFrame()
    
    return {"dataframe": df, "column_names": column_names}

def app():    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        st.markdown("Use the buttons below to navigate to different sections of the app.")
        if st.button("Upload Data"):
            st.session_state['page'] = 'Upload Data'
        if st.button("Power Forecasting"):
            st.session_state['page'] = 'Power Forecasting'
        if st.button("Electricity Theft Detection"):
            st.session_state['page'] = 'Electricity Theft Detection'
        if st.button("Energy Optimization"):
            st.session_state['page'] = 'Energy Optimization'

    # Main Page Content
    st.title("Appliance Load Data Analysis")
    st.markdown("""
        Welcome to the Appliance Load Data Analysis app. This application allows you to upload and analyze appliance load data, 
        and navigate through different analysis tools such as Power Forecasting, Electricity Theft Detection, and Energy Optimization.
        Follow the instructions below to get started.
    """)
    
    if st.session_state.get('page', 'Upload Data') == 'Upload Data':
        st.header("Upload Appliance Load Data")
        
        st.markdown("### Instructions:")
        st.markdown("""
            1. Ensure your files are correctly structured and named.
            2. Click the "Fetch Data" button to load data from the provided GitHub repository.
        """)
        
        # URLs of the files in your GitHub repository
        base_url = "https://raw.githubusercontent.com/opeyemiorugun/finalyearproject/master/"  # Adjust the URL based on your repository structure
        label_file_url = base_url + "labels.dat"
        weather_file_url = base_url + "weather.csv"
        csv_files_urls = [base_url + f"channel_{i}.dat" for i in range(2, 26)]  # Adjust the filenames as per your repository

        if st.button("Fetch Data"):
            st.spinner("Fetching data...")
            label_file, label_filename = fetch_github_file(label_file_url)
            if label_file:
                uploaded_files = {'label_file': label_file, 'csv_files': [fetch_github_file(url) for url in csv_files_urls]}
                if all(file[0] for file in uploaded_files['csv_files']):
                    data = load_data(uploaded_files)
                    if not data["dataframe"].empty:
                        st.success("Data Loaded Successfully")
                        with st.expander("Preview Data"):
                            st.write(data["dataframe"].head())
                        
                        # Fetch the weather file from GitHub
                        weather_file, weather_filename = fetch_github_file(weather_file_url)
                        if weather_file:
                            try:
                                weather_csv = pd.read_csv(weather_file)
                                st.success("Weather Data Loaded Successfully")
                                with st.expander("Preview Weather Data"):
                                    st.write(weather_csv.head())

                                # Store data in session state
                                st.session_state['uploaded_data'] = data["dataframe"]
                                st.session_state['column_names'] = data["column_names"]
                                st.session_state["weather_data"] = weather_csv
                            except Exception as e:
                                st.error(f"Error reading weather file: {e}")
                        else:
                            st.warning("Please upload the weather file.")
                    else:
                        st.error("No valid data found.")
                else:
                    st.error("No files uploaded.")
            else:
                st.error("Please upload the label file.")
    else:
        st.write("Select a section from the sidebar to get started.")

if __name__ == "__main__":
    app()
