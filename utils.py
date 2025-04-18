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
      # Time Slot Suggestion
    if avg_temp <= 33:
        time_slots = ["6:00 AM - 10:00 AM", "5:30 PM - 8:00 PM"]
    elif avg_temp <= 37:
        time_slots = ["6:00 AM - 8:00 AM", "6:30 PM - 7:30 PM"]
    else:
        time_slots = ["⚠️ Stay Indoors - Too Hot Outside"]