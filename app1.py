import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Linear Regression Weather App",
    page_icon="🌤️",
    layout="centered"
)

st.title("🌡️ Weather Prediction")
st.write("Linear Regression Model")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("weather.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
st.success("✅ Linear Regression Model Loaded")

# -----------------------------
# Inputs
# -----------------------------
st.subheader("Enter Input Values")

hours_sunlight = st.number_input(
    "Hours of Sunlight",
    min_value=0.0,
    step=0.1
)

humidity_level = st.number_input(
    "Humidity Level (%)",
    min_value=0.0,
    max_value=100.0,
    step=1.0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Temperature"):
    input_data = np.array([[hours_sunlight, humidity_level]])
    prediction = model.predict(input_data)

    st.success(
        f"🌡️ Predicted Daily Temperature: {prediction[0]:.2f} °C"
    )
