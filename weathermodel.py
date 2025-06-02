# weather_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv("wheater and location.csv")

# Encode target label (Crop_Type) and categorical feature (Adaptation_Strategies)
le_crop = LabelEncoder()
le_strategy = LabelEncoder()

df["Crop_Type"] = le_crop.fit_transform(df["Crop_Type"])
df["Adaptation_Strategies"] = le_strategy.fit_transform(df["Adaptation_Strategies"])

# Select features
features = [
    "Average_Temperature_C",
    "Total_Precipitation_mm",
    "CO2_Emissions_MT",
    "Extreme_Weather_Events",
    "Irrigation_Access_%",
    "Pesticide_Use_KG_per_HA",
    "Fertilizer_Use_KG_per_HA",
    "Soil_Health_Index",
    "Adaptation_Strategies"
]
X = df[features]
y = df["Crop_Type"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "model_weather_rf.pkl")
joblib.dump(le_crop, "le_crop_weather.pkl")
joblib.dump(le_strategy, "le_strategy.pkl")

print("✅ Weather-based crop model trained and saved!")
