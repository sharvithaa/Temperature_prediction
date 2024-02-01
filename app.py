import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
with open("weather_model.joblib", "rb") as model_file:
    model = joblib.load(model_file)

# Streamlit UI
st.title("Temprature Prediction")

# Input form
st.sidebar.header("User Input Features")

def user_input_features():
    latitude = st.sidebar.number_input("Enter Latitude", -90.0, 90.0, key='latitude')
    longitude = st.sidebar.number_input("Enter Longitude", -180.0, 180.0, key='longitude')
    wind_kph = st.sidebar.number_input("Enter Wind Speed (kph)", min_value=0.0, value=10.0)
    wind_degree = st.sidebar.number_input("Enter Wind Degree", min_value=0, max_value=360, value=180)
    pressure_mb = st.sidebar.number_input("Enter Pressure (mb)", min_value=0.0, value=1010.0)
    precip_in = st.sidebar.number_input("Enter Precipitation (in)", min_value=0.0, value=0.0)
    humidity = st.sidebar.number_input("Enter Humidity (%)", min_value=0, max_value=100, value=50)
    cloud = st.sidebar.number_input("Enter Cloud Cover (%)", min_value=0, max_value=100, value=25)

     # Include 'PassengerId' with user input
    data = {
       'latitude': [latitude],
    'longitude': [longitude],
    'wind_kph': [wind_kph],
    'wind_degree': [wind_degree],
    'pressure_mb': [pressure_mb],
    'precip_in': [precip_in],
    'humidity': [humidity],
    'cloud': [cloud]
    }

    features = pd.DataFrame(data, index=[0])

    # Ensure column names match those used during training
    features = features[['latitude','longitude','wind_kph','wind_degree','pressure_mb','precip_in','humidity','cloud']]

    return features  # Exclude 'PassengerId'




input_df = user_input_features()

# Show the input data
st.subheader("User Input:")
st.write(input_df)

# Make predictions
# Exclude 'PassengerId' before making predictions
prediction = model.predict(input_df)

# Display the prediction
st.subheader("Prediction")
st.write("Predicted Temperature:",round(prediction[0],2), "°C")


