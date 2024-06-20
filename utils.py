import pandas as pd
import joblib
import numpy as np
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import xgboost as xgb
import lightgbm as lgm

def load_model(appliances):
    models = {}
    for appliance in appliances:
        filepath = f"../interactive_app/models/{appliance}_model.joblib"
        try:
            models[appliance] = joblib.load(filepath)
        except OSError as e:
            st.error(f"Error loading model for {appliance}: {e}")
    return models

@st.cache_data
def preprocess_data(data, column_names):
    def handle_missing_values(df):
        resampled_df = df.resample('6s').mean()
        for col in resampled_df.columns:
            resampled_df[col] = pd.to_numeric(resampled_df[col], errors='coerce')
        final_df = resampled_df.interpolate(method='linear')
        na_columns = final_df.columns[final_df.isna().any()].tolist()
        for col in na_columns:
            final_df[col] = final_df[col].fillna(0)
        return final_df

    def handle_standby_values(df, column_names):
        final_standby = df
        for column in column_names:
            state_column = f'{column}_state'
            final_standby[state_column] = final_standby[column].apply(lambda x: 'off' if x == 0 else 'unknown')
        for column in column_names:
            state_column = f'{column}_state'
            non_zero_values = final_standby[final_standby[column] != 0][column].values.reshape(-1, 1)
            if len(non_zero_values) > 1:
                kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(non_zero_values)
                standby_cluster = np.argmin(kmeans.cluster_centers_)
                active_cluster = 1 - standby_cluster
                cluster_mapping = {standby_cluster: 'standby', active_cluster: 'active'}
                final_standby.loc[final_standby[column] != 0, state_column] = kmeans.labels_
                final_standby[state_column] = final_standby[state_column].replace(cluster_mapping)
            else:
                if len(non_zero_values) == 1:
                    final_standby.loc[final_standby[column] != 0, state_column] = 'active'
        standby_active = {"kettle": 2000, "fridge_freezer": 50, "dishwasher": 10, "microwave": 200, "washer_dryer": 20}
        for key in standby_active.keys():
            if key in column_names:
                final_standby[f"{key}_state"] = np.where(final_standby[key] > standby_active[key], "active", final_standby[f"{key}_state"])
        return final_standby

    def handle_outliers(df, column_names, lower_quantile=0.01, upper_quantile=0.99):
        final_df_c = df.copy()
        final_df_c["timestamp"] = df.index
        df_cleaned = final_df_c.copy()
        for column in column_names[1:]:
            if column != 'timestamp':
                appliance_data = final_df_c[column]
                if column != "aggregate":
                    appliance_state = final_df_c[f"{column}_state"]
                    active_data = appliance_data[appliance_state == 'active']
                else:
                    active_data = appliance_data[appliance_data > 0]
                lower_bound = active_data.quantile(lower_quantile)
                upper_bound = active_data.quantile(upper_quantile)
                if column != 'aggregate':
                    cleaned_data = appliance_data[(appliance_state != 'active') | 
                                            ((appliance_data >= lower_bound) & (appliance_data <= upper_bound))]
                else:
                    cleaned_data = appliance_data[(appliance_data == 0) | 
                                                ((appliance_data >= lower_bound) & (appliance_data <= upper_bound))]
                df_cleaned[column] = cleaned_data   
        df_cleaned.dropna(inplace=True)
        df_p = final_df_c.loc[df_cleaned.index]
        return df_p

    def feature_engineering(df, column_names):
        df_p = df.copy()
        new_columns = {}
        for col in column_names:
            standby = df_p.loc[df_p[f'{col}_state'] == 'standby', col]
            avg_standby = standby.mean()
            new_columns[f'{col}_avg_standby'] = [avg_standby] * len(df_p)
            total_power_consumption = df_p[col].sum()
            total_standby_consumption = standby.sum()
            standby_ratio = total_standby_consumption / total_power_consumption if total_power_consumption != 0 else 0
            new_columns[f'{col}_standby_ratio'] = [standby_ratio] * len(df_p)
            deviation = df_p[col] - avg_standby
            new_columns[f'{col}_deviation'] = deviation
        new_columns_df = pd.DataFrame(new_columns, index=df_p.index)
        df_p = pd.concat([df_p, new_columns_df], axis=1)
        df_p = df_p.copy()
        df_p = pd.get_dummies(df_p, drop_first=False)
        temp_hum = st.session_state["weather_data"]
        temp_hum['datetime'] = pd.to_datetime(temp_hum.YEAR.astype(str) + '-' + temp_hum.MO.astype(str) +'-'
                                            + temp_hum.DY.astype(str) + ' ' + temp_hum.HR.astype(str) + ":00:00" )
        temp_hum.set_index('datetime', inplace=True)
        temp_hum.drop(['YEAR', 'MO', 'DY', 'HR'], axis=1, inplace=True)
        weather = temp_hum.resample('6s').interpolate()
        data = df_p.join(weather)
        data['datetime'] = data.index
        data['hour'] = data['datetime'].dt.hour
        data['dayofweek'] = data['datetime'].dt.dayofweek
        data['quarter'] = data['datetime'].dt.quarter
        data['year'] = data['datetime'].dt.year
        data['dayofyear'] = data['datetime'].dt.dayofyear
        data['dayofmonth'] = data['datetime'].dt.day
        data['month'] = data['datetime'].dt.month
        data.drop(["datetime"], axis=1, inplace=True)
        lagged_features = []
        for col in column_names:
            if col not in ['dayofweek','datetime', 'T2M', 'RH2M', 'month', 'year', 'hour', 'quarter', 'dayofyear', 'dayofmonth']:
                lagged_features.append(df[col].shift(1).rename(f'{col}_prev_day'))
        lagged_df = pd.concat([data] + lagged_features, axis=1)
        fourier = CalendarFourier(freq="YE", order=10)
        dp = DeterministicProcess(
            index=data.index,
            constant=True,
            order=1,
            seasonal=True,
            period=7,
            additional_terms=[fourier],
            drop=True,
        )
        det_d = dp.in_sample()
        print(det_d.columns)
        combined = pd.merge(lagged_df, det_d, left_index=True, right_index=True)
        output = {"main": combined, "extra": det_d}
        return output

    df = data
    data = handle_missing_values(df)
    data = handle_standby_values(data, column_names)
    data = handle_outliers(data, column_names)
    data = feature_engineering(data, column_names)
    return data

def train_split_test(data, app):
    # combined = data["main"]
    # det_d = data["extra"]
    # time_features = ['dayofweek', 'T2M', 'RH2M', 'month', 'year', 'hour', 'quarter', 'dayofyear', 'dayofmonth']
    # det_features = [col for col in det_d.columns]
    # appliance_cols = [col for col in combined.columns if app in col]
    # keep_cols = time_features + det_features + appliance_cols
    # appliance_df = combined[keep_cols]
    # test_df = appliance_df.loc["2014-10-01 12:00:00":]
    X_test, y_test = data.drop(columns=[app]), data[app]
    test = {"X_test": X_test, "y_test": y_test}
    return test

def forecast_power(data, selected_appliances, models, horizon):
    # Preprocess the historical data first
    combined = data["main"]
    # Generate future dates for the forecast period
    future_dates = pd.date_range(start=combined.index[-1] + pd.Timedelta(seconds=6), periods=horizon, freq='6S')
    
    forecast_df = pd.DataFrame(index=future_dates)

    # Loop through each selected appliance to generate forecasts
    for appliance in selected_appliances:
        model = models[appliance]
        
        # Create an empty DataFrame for future dates with the same columns as pre_processed_data
        future_df = pd.DataFrame(index=future_dates, columns=combined.columns)
        
        # Concatenate historical data with future DataFrame
        full_df = pd.concat([combined, future_df])
        
        # Handle empty rows in the concatenated DataFrame
        full_df = full_df.fillna(method='ffill').fillna(method='bfill')

        # Prepare the data for prediction
        app_specific = train_split_test(full_df, appliance)
        
        # Generate predictions
        forecast = model.predict(app_specific["X_test"])
        
        # Store the forecast values in the forecast_df
        forecast_df[appliance] = forecast[:horizon]

    return forecast_df

def calculate_metrics(actuals, predictions):
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    return mae, rmse

def plot_forecast_vs_actual(actuals, predictions, selected_appliances):
    for appliance in selected_appliances:
        plt.figure(figsize=(10, 6))
        plt.plot(actuals.index, actuals[appliance], label='Actual')
        plt.plot(predictions.index, predictions[appliance], label='Forecast')
        plt.title(f'Forecast vs Actual for {appliance}')
        plt.legend()
        st.pyplot(plt)

def plot_feature_importance(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots()
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(feature_names[indices], rotation=90)
    st.pyplot(fig)

def detect_theft(data, model):
    predictions = model.predict(data)
    return predictions

def plot_theft_detection(data, predictions):
    fig, ax = plt.subplots()
    ax.plot(data.index, data['power_usage'], label="Power Usage")
    ax.plot(data.index, predictions, label="Theft Prediction")
    ax.legend()
    st.pyplot(fig)

def calculate_total_forecasted_consumption(forecast_df, row):
    total_forecasted_consumption = 0
    for col in forecast_df.columns:
        if col.endswith('_forecast'):
            total_forecasted_consumption += row[col]
    return total_forecasted_consumption

def reschedule_appliances_based_on_forecast(df, forecast_df, peak_threshold):
    changes_log = []
    for index, row in forecast_df.iterrows():
        if row['total_forecasted_consumption'] > peak_threshold:
            for col in forecast_df.columns:
                if col.endswith('_forecast'):
                    appliance = col.replace('_forecast', '')
                    if df.at[index, f'{appliance}_state'] == 'active':
                        non_peak_period = forecast_df[(forecast_df['datetime'] > row['datetime']) & (forecast_df['total_forecasted_consumption'] <= peak_threshold)].head(1)
                        if not non_peak_period.empty:
                            next_non_peak_index = non_peak_period.index[0]
                            df.at[next_non_peak_index, f'{appliance}_state'] = 'active'
                            df.at[index, f'{appliance}_state'] = 'off'
                            changes_log.append((appliance, row['datetime'], row['total_forecasted_consumption']))
    changes_log_df = pd.DataFrame(changes_log, columns=['Appliance', 'Original_Time', 'Forecasted_Consumption'])
    return df, changes_log_df

def minimize_standby_consumption(df, standby_threshold_hours=1):
    standby_threshold = timedelta(hours=standby_threshold_hours)
    changes_log = []
    for index, row in df.iterrows():
        for col in df.columns:
            if col.endswith('_standby') and row[col] == True:
                appliance = col.replace('_standby', '')
                previous_timestamps = df.index.get_loc(row.name) - pd.RangeIndex(1, 11)
                previous_states = df.iloc[previous_timestamps]
                if not previous_states.empty:
                    last_state = previous_states.iloc[-1][appliance]
                    last_time = previous_states.index[-1]
                    if last_state == True and (row.name - last_time) > standby_threshold:
                        df.at[index, appliance] = 'off'
                        changes_log.append((appliance, row.name))
    changes_log_df = pd.DataFrame(changes_log, columns=['Appliance', 'Time_Changed'])
    return df, changes_log_df

def optimize_energy_usage(actual_data, forecasted_data, goals, peak_threshold=1000, standby_threshold_hours=1):
    actual_data['datetime'] = pd.to_datetime(actual_data['datetime'])
    forecasted_data['datetime'] = pd.to_datetime(forecasted_data['datetime'])
    forecasted_data['total_forecasted_consumption'] = forecasted_data.apply(lambda row: calculate_total_forecasted_consumption(forecasted_data, row), axis=1)
    changes_log_df = pd.DataFrame()
    if 'Minimize Standby Consumption' in goals:
        df, log_df = minimize_standby_consumption(actual_data, standby_threshold_hours)
        changes_log_df = pd.concat([changes_log_df, log_df])
    if 'Load Balancing' in goals:
        df, log_df = reschedule_appliances_based_on_forecast(actual_data, forecasted_data, peak_threshold)
        changes_log_df = pd.concat([changes_log_df, log_df])
    return df, changes_log_df

def plot_optimization_results(forecasted_data, optimized_data, selected_appliances):
    for appliance in selected_appliances:
        fig, ax = plt.subplots()
        ax.plot(forecasted_data.index, forecasted_data[appliance], label="Forecasted")
        ax.plot(optimized_data.index, optimized_data[appliance], label="Optimized")
        ax.legend()
        st.pyplot(fig)

def minimize_standby(df, column_names, standby_threshold=timedelta(hours=1)):
    changes_log = []
    for index, row in df.iterrows():
        for col in column_names:
            standby = f'{col}_standby'
            if row[standby] == True:
                appliance = col
                previous_timestamps = df.index.get_loc(row.name) - range(1, 11)
                previous_states = df.iloc[previous_timestamps]
                if not previous_states.empty:
                    last_state = previous_states.iloc[-1][col]
                    last_time = previous_states.iloc[-1].name
                    if last_state == True and (row.name - last_time) > standby_threshold:
                        df.at[index, col] = 'off'
                        changes_log.append((appliance, row.name))
    return df, changes_log

def load_balancing(df, column_names):
    def calculate_total_consumption(row):
        total_consumption = 0
        for col in column_names:
            active = f'{col}_active'
            if row[active] == True:
                total_consumption += row[col]
        return total_consumption

    df['total_consumption'] = df.apply(calculate_total_consumption, axis=1)
    peak_threshold = df['total_consumption'].quantile(0.90)
    df['is_peak'] = df['total_consumption'] > peak_threshold

    def reschedule_appliances(row):
        if row['is_peak']:
            for col in column_names:
                active = f'{col}_active'
                if row[active] == True:
                    non_peak_period = df[(df['datetime'] > row['datetime']) & (df['is_peak'] == False)].head(1)
                    if not non_peak_period.empty:
                        next_non_peak_index = non_peak_period.index[0]
                        df.at[next_non_peak_index, col] = 'active'
                        df.at[row.name, col] = 'off'
        return row

    df = df.apply(reschedule_appliances, axis=1)
    return df

def lifetime_extension(df, usage_guidelines):
    changes_log = []
    def check_and_adjust_usage(row):
        for appliance, guidelines in usage_guidelines.items():
            state_col = f'{appliance}_state'
            if row[state_col] == 'active':
                start_of_day = row['datetime'].replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = start_of_day + timedelta(days=1)
                daily_usage = df[(df['datetime'] >= start_of_day) & (df['datetime'] < end_of_day) & (df[state_col] == 'active')].shape[0]
                if daily_usage > guidelines['max_daily_usage']:
                    df.at[row.name, state_col] = 'off'
                    changes_log.append((appliance, row['datetime'], 'exceeded max daily usage'))
                last_active_time = df[(df['datetime'] < row['datetime']) & (df[state_col] == 'active')].tail(1)['datetime']
                if not last_active_time.empty:
                    last_active_time = last_active_time.values[0]
                    if row['datetime'] - last_active_time < guidelines['cool_down_period']:
                        df.at[row.name, state_col] = 'off'
                        changes_log.append((appliance, row['datetime'], 'insufficient cool down period'))
        return row

    df = df.apply(check_and_adjust_usage, axis=1)
    return df, changes_log

def minimize_energy_cost(df, column_names, faux_time_of_use_rates, rate_schedule):
    def get_rate(hour):
        for period in rate_schedule['peak']:
            if period[0] <= hour < period[1]:
                return faux_time_of_use_rates['peak']
        for period in rate_schedule['off_peak']:
            if period[0] <= hour < period[1]:
                return faux_time_of_use_rates['off_peak']
        for period in rate_schedule['super_off_peak']:
            if period[0] <= hour < period[1]:
                return faux_time_of_use_rates['super_off_peak']
        return faux_time_of_use_rates['off_peak']

    def calculate_energy_cost(row):
        total_cost = 0
        hour = row['datetime'].hour
        rate = get_rate(hour)
        for col in column_names:
            if row[col + '_state'] == 'active':
                power_consumption = row[col]
                total_cost += power_consumption * rate
        return total_cost

    df['energy_cost'] = df.apply(calculate_energy_cost, axis=1)

    def reschedule_appliances(row):
        hour = row['datetime'].hour
        rate = get_rate(hour)
        if rate == faux_time_of_use_rates['peak']:
            for col in column_names:
                if row[col + '_state'] == 'active':
                    non_peak_period = df[(df['datetime'] > row['datetime']) & (get_rate(df['datetime'].dt.hour) != faux_time_of_use_rates['peak'])].head(1)
                    if not non_peak_period.empty:
                        next_non_peak_index = non_peak_period.index[0]
                        df.at[next_non_peak_index, col] = 'active'
                        df.at[row.name, col] = 'off'
        return row

    df = df.apply(reschedule_appliances, axis=1)
    return df

def minimize_noise(df, noise_categories, quiet_hours_start=22, quiet_hours_end=7):
    def reschedule_noisy_appliances(row):
        hour = row['datetime'].hour
        if quiet_hours_start <= hour or hour < quiet_hours_end:
            for appliance, noise_level in noise_categories.items():
                state_col = f'{appliance}_state'
                if row[state_col] == 'active' and noise_level == 'noisy':
                    next_non_quiet_period = df[(df['datetime'] > row['datetime']) & ((df['datetime'].dt.hour < quiet_hours_start) & (df['datetime'].dt.hour >= quiet_hours_end))].head(1)
                    if not next_non_quiet_period.empty:
                        next_non_quiet_index = next_non_quiet_period.index[0]
                        df.at[next_non_quiet_index, state_col] = 'active'
                        df.at[row.name, state_col] = 'off'
        return row

    df = df.apply(reschedule_noisy_appliances, axis=1)
    return df
