import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import joblib

# Load your dataset
data = pd.read_csv("IndianWeatherRepository.csv")
data.drop(['temperature_fahrenheit','wind_mph','pressure_in','precip_mm','feels_like_fahrenheit','visibility_km','gust_mph'], axis=1, inplace=True)

numeric_columns = data.select_dtypes(include=[np.number]).columns
corr_matrix = data[numeric_columns].corr(method="kendall")
sorted_corr_mat = corr_matrix.abs().unstack().sort_values()
sorted_corr_mat = corr_matrix.drop(corr_matrix[corr_matrix > 0.95].index)
sorted_corr_mat=corr_matrix.drop(corr_matrix[corr_matrix<0.05].index)

# Separate features and target variable
temperature_data = data['temperature_celsius']
temperature_factors = data[['latitude','longitude','wind_kph','wind_degree','pressure_mb','precip_in','humidity','cloud']]

x_train, x_test, y_train, y_test = train_test_split(temperature_factors, temperature_data, test_size = 0.3, random_state = 0)



# Train the model using the entire dataset
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

print("Accuracy : ",r2_score(y_test,y_pred)*100)

joblib.dump(regressor, 'weather_model.joblib')