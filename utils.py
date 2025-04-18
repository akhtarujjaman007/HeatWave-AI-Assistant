# utils.py
import pandas as pd
import joblib

# Load trained model
model = joblib.load("heatwave_model.pkl")
features = ['AvgTemp', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm']

def recommend(temp, humidity, pressure, cloud):
    avg_temp = temp
    hi_input = pd.DataFrame([[avg_temp, humidity, pressure, cloud]], columns=features)
    is_heatwave = model.predict(hi_input)[0]