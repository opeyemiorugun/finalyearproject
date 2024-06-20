import streamlit as st
import pandas as pd
import requests
from io import StringIO

def fetch_github_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        return StringIO(response.text), url.split('/')[-1]  # Return file-like object and filename
    else:
        st.error(f"Failed to fetch file from {url}")
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
        st.write(f"Appliance number from file: {appliance_number}")
        st.write(f"Label dictionary: {label_dict}")

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
    st.title("Upload Appliance Load Data")

    # URLs of the files in your GitHub repository
    base_url = "https://raw.githubusercontent.com/opeyemiorugun/Empower/master/data/"  # Adjust the URL based on your repository structure
    label_file_url = base_url + "labels.dat"
    weather_file_url = base_url + "weather.csv"
    csv_files_urls = [base_url +  "house_5/" + f"channel_{i}.dat" for i in range(2, 26)]  # Adjust the filenames as per your repository


    # Fetch the label file from GitHub
    label_file, label_filename = fetch_github_file(label_file_url)
    if label_file:
        uploaded_files = {'label_file': label_file, 'csv_files': [fetch_github_file(url) for url in csv_files_urls]}
        if all(file[0] for file in uploaded_files['csv_files']):
            data = load_data(uploaded_files)
            if not data["dataframe"].empty:
                st.write("Data Loaded Successfully")
                st.write(data["dataframe"].head())
                
                # Fetch the weather file from GitHub
                weather_file, weather_filename = fetch_github_file(weather_file_url)
                if weather_file:
                    try:
                        weather_csv = pd.read_csv(weather_file)
                        st.write("Weather Data Loaded Successfully")
                        st.write(weather_csv.head())

                        # Store data in session state
                        st.session_state['uploaded_data'] = data["dataframe"]
                        st.session_state['column_names'] = data["column_names"]
                        st.session_state["weather_data"] = weather_csv

                        # Navigation buttons
                        if st.button("Go to Power Forecasting"):
                            st.session_state['page'] = 'Power Forecasting'
                        if st.button("Go to Electricity Theft Detection"):
                            st.session_state['page'] = 'Electricity Theft Detection'
                        if st.button("Go to Energy Optimization"):
                            st.session_state['page'] = 'Energy Optimization'
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

if __name__ == "__main__":
    app()
