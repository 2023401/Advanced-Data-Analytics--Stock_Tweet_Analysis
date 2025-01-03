import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    forecast_results = {
        "AAPL": {
            "Actual": [121.09999847, 121.09999847, 121.09999847, 121.09999847, 121.09999847, 121.09999847, 121.09999847],
            "LSTM": [122.61004, 122.14471, 121.83347],
            "SARIMAX": [121.12683951, 121.11995742, 121.12699803],
        },
        "ABNB": {
            "Actual": [163.02000427, 163.19000244, 158.00999451, 154.83999634, 149.0, 150.0, 146.80000305],
            "LSTM": [138.33107, 139.29211, 139.41121],
            "SARIMAX": [163.38443732, 168.65673712, 177.40868138],
        },
        "AMZN": {
            "Actual": [172.1815033, 172.1815033, 172.1815033, 168.1855011, 168.1855011, 166.93249512, 163.63549805],
            "LSTM": [157.17392, 160.62527, 165.90346],
            "SARIMAX": [173.11743525, 172.73844314, 172.09791438],
        },
        "BABA": {
            "Actual": [264.42999268, 264.42999268, 255.83000183, 255.83000183, 255.83000183, 255.83000183, 256.17999268],
            "LSTM": [267.28174, 264.98492, 260.9552],
            "SARIMAX": [255.44286804, 256.30884546, 256.3414691],
        },
        "BAC": {
            "Actual": [26.02000046, 25.65999985, 25.28000069, 25.23999977, 23.62000084, 27.65999985, 26.69000053],
            "LSTM": [23.232536, 23.227331, 23.260733],
            "SARIMAX": [25.8231212, 25.44669637, 25.87980452],
        },
    }
    return forecast_results

# Load forecast data
forecast_results = load_data()

# Sidebar Controls
st.sidebar.title("Forecast Dashboard")
selected_stock = st.sidebar.selectbox("Select Stock", list(forecast_results.keys()))
selected_method = st.sidebar.radio("Select Method", ["LSTM", "SARIMAX"])
forecast_horizon = st.sidebar.slider("Select Forecast Horizon", min_value=1, max_value=7, value=3)

# Main Content
st.title(f"Forecast Dashboard for {selected_stock}")

# Fetch data for selected stock and method
actual = forecast_results[selected_stock]["Actual"]
predicted = forecast_results[selected_stock][selected_method]

# Filter data based on forecast horizon
actual_filtered = actual[:forecast_horizon]
predicted_filtered = predicted[:forecast_horizon]

# Line Chart
st.subheader("Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(actual_filtered)), actual_filtered, label="Actual", color="blue")
ax.plot(range(len(predicted_filtered)), predicted_filtered, label=selected_method, color="red")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
ax.set_title(f"{selected_method} Forecast for {selected_stock}")
ax.legend()
st.pyplot(fig)

# Error Metrics
st.subheader("Error Metrics")
mae = round(sum(abs(a - p) for a, p in zip(actual_filtered, predicted_filtered)) / forecast_horizon, 2)
rmse = round((sum((a - p) ** 2 for a, p in zip(actual_filtered, predicted_filtered)) / forecast_horizon) ** 0.5, 2)
st.write(f"**Mean Absolute Error (MAE):** {mae}")
st.write(f"**Root Mean Square Error (RMSE):** {rmse}")

# Download Option
st.subheader("Download Predictions")
download_data = pd.DataFrame({
    "Day": range(1, forecast_horizon + 1),
    "Actual": actual_filtered,
    f"{selected_method} Prediction": predicted_filtered
})
st.download_button(
    label="Download Predictions as CSV",
    data=download_data.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_stock}_{selected_method}_forecast.csv",
    mime="text/csv"
)
