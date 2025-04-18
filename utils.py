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

    # Safe Age Groups
    age_groups = {
        "Children (0-12)": 32,
        "Teens (13-19)": 35,
        "Adults (20-59)": 38,
        "Elderly (60+)": 30
    }
    safe_ages = [group for group, max_temp in age_groups.items() if avg_temp <= max_temp]

    return {
        "Heatwave": bool(is_heatwave),
        "Safe Time Slots": time_slots,
        "Safe Age Groups": safe_ages
    }
