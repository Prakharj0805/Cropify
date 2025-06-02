import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("soildata.csv")

# Encode categorical columns
le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_fert = LabelEncoder()

df["Soil Type"] = le_soil.fit_transform(df["Soil Type"])
df["Fertilizer Name"] = le_fert.fit_transform(df["Fertilizer Name"])
df["Crop Type"] = le_crop.fit_transform(df["Crop Type"])

# Save label encoders
joblib.dump(le_crop, "le_crop.pkl")
joblib.dump(le_soil, "le_soil.pkl")
joblib.dump(le_fert, "le_fert.pkl")

# Prepare data
X = df.drop(columns=["Crop Type"])
y = df["Crop Type"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "soil_model.pkl")
print("✅ Soil model trained and saved.")
