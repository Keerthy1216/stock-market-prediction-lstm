# app.py

import os

# -----------------------------
# Environment Setup
# -----------------------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input


# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.title("Google Stock Price Prediction (LSTM)")

st.sidebar.title("Stock Prediction")
st.sidebar.write("Model: LSTM")
st.sidebar.write("Sequence Length: 60 days")
st.sidebar.write("Framework: TensorFlow + Streamlit")

# -----------------------------
# File Paths
# -----------------------------
DATA_PATH = "dataset/google.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())


# -----------------------------
# Stock Price Chart
# -----------------------------
st.subheader("Stock Close Price Chart")
st.line_chart(data["Close"])

# -----------------------------
# Data Preprocessing
# -----------------------------
close_prices = data[["Close"]].values

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# -----------------------------
# Create Sequences
# -----------------------------
def create_sequences(data, seq_length=60):

    X, y = [], []

    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i,0])
        y.append(data[i,0])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


X, y = create_sequences(scaled_data)


# -----------------------------
# Train Model (if not exists)
# -----------------------------
if not os.path.exists(MODEL_PATH):

    st.subheader("Training LSTM Model...")

    model = Sequential([
        Input(shape=(60,1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, batch_size=32)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    st.success("Model trained and saved!")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mse")
    return model

model = load_trained_model()

st.info("Loaded trained model.")


# -----------------------------
# Model Evaluation
# -----------------------------
train_pred = model.predict(X)

train_pred = scaler.inverse_transform(train_pred)
y_actual = scaler.inverse_transform(y.reshape(-1,1))

mae = mean_absolute_error(y_actual, train_pred)
rmse = np.sqrt(mean_squared_error(y_actual, train_pred))

st.subheader("Model Performance")

st.write(f"MAE (Mean Absolute Error): {mae:.4f}")
st.write(f"RMSE (Root Mean Squared Error): {rmse:.4f}")


# -----------------------------
# Next Day Prediction
# -----------------------------
st.subheader("Predicted Next Close Price")

last_60 = scaled_data[-60:]

X_test = np.array([last_60]).reshape((1,60,1))

predicted_scaled = model.predict(X_test)

predicted_price = scaler.inverse_transform(predicted_scaled)

st.success(f"${predicted_price[0][0]:.4f}")

# -----------------------------
# Multi-Day Future Prediction
# -----------------------------
days = st.slider("Select number of future days to predict",1,10,1)

future_predictions = []

current_input = last_60.copy()

for _ in range(days):

    X_future = current_input.reshape((1,60,1))

    pred = model.predict(X_future)[0][0]

    future_predictions.append(pred)

    current_input = np.append(current_input[1:], [[pred]], axis=0)

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1,1)
)

st.subheader("Future Predictions")

for i, price in enumerate(future_predictions):
    st.write(f"Day {i+1}: ${price[0]:.2f}")

# -----------------------------
# Matplotlib Chart
# -----------------------------
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(data["Close"][-30:].values, label="Actual Price")

ax.plot(
    [29,30],
    [data["Close"].iloc[-1], predicted_price[0][0]],
    "ro-",
    label="Predicted Next Day"
)

ax.set_title("Stock Price Prediction")
ax.set_xlabel("Days")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

# -----------------------------
# Interactive Plotly Chart
# -----------------------------
st.subheader("Interactive Stock Chart")

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        y=data["Close"][-30:],
        mode="lines",
        name="Actual Price"
    )
)

fig2.add_trace(
    go.Scatter(
        x=[29],
        y=[predicted_price[0][0]],
        mode="markers",
        name="Predicted Price"
    )
)

fig2.update_layout(
    title="Stock Price Prediction",
    xaxis_title="Days",
    yaxis_title="Price"
)

st.plotly_chart(fig2)