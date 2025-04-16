# weather_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load and preprocess dataset
df = pd.read_csv("Weather_Data.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')
df.dropna(inplace=True)
df['AvgTemp'] = (df['Temp9am'] + df['Temp3pm']) / 2
df['Heatwave'] = df['MaxTemp'] > 38

# Select features and labels
features = ['AvgTemp', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm']
X = df[features]
y = df['Heatwave']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "heatwave_model.pkl")
