from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load soil model and encoders
soil_model = joblib.load('models/soil_model.pkl')
le_crop_soil = joblib.load('models/le_crop.pkl')
le_soil = joblib.load('models/le_soil.pkl')
le_fert = joblib.load('models/le_fert.pkl')

# Load weather model and encoders
weather_model = joblib.load('models/model_weather_rf.pkl')
le_crop_weather = joblib.load('models/le_crop_weather.pkl')
le_strategy = joblib.load('models/le_strategy.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict/soil', methods=['POST'])
def predict_soil():
    try:
        data = request.json

        # Create DataFrame with ALL required input features
        input_df = pd.DataFrame({
            'Temperature': [float(data['temperature'])],
            'Humidity': [float(data['humidity'])],
            'Moisture': [float(data['moisture'])],
            'Soil Type': [le_soil.transform([data['soil_type']])[0]],
            'Nitrogen': [float(data['nitrogen'])],
            'Potassium': [float(data['potassium'])],
            'Phosphorous': [float(data['phosphorous'])],
            'Fertilizer Name': [le_fert.transform([data['fertilizer_name']])[0]]
        })

        # Make prediction
        prediction = soil_model.predict(input_df)[0]
        crop_name = le_crop_soil.inverse_transform([prediction])[0]
        confidence = float(np.max(soil_model.predict_proba(input_df)[0]))

        return jsonify({
            'success': True,
            'model': 'soil',
            'prediction': crop_name,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/predict/weather', methods=['POST'])
def predict_weather():
    try:
        data = request.json

        # Create DataFrame with the input data
        input_df = pd.DataFrame({
            'Average_Temperature_C': [float(data['avg_temp'])],
            'Total_Precipitation_mm': [float(data['precipitation'])],
            'CO2_Emissions_MT': [float(data['co2_emissions'])],
            'Extreme_Weather_Events': [int(data['extreme_weather'])],
            'Irrigation_Access_%': [float(data['irrigation_access'])],
            'Pesticide_Use_KG_per_HA': [float(data['pesticide_use'])],
            'Fertilizer_Use_KG_per_HA': [float(data['fertilizer_use'])],
            'Soil_Health_Index': [float(data['soil_health'])],
            'Adaptation_Strategies': [le_strategy.transform([data['adaptation_strategy']])[0]]
        })

        # Make prediction
        prediction = weather_model.predict(input_df)[0]
        crop_name = le_crop_weather.inverse_transform([prediction])[0]
        confidence = float(np.max(weather_model.predict_proba(input_df)[0]))

        return jsonify({
            'success': True,
            'model': 'weather',
            'prediction': crop_name,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/predict/multi', methods=['POST'])
def predict_multi():
    try:
        data = request.json

        # Soil model input - include ALL required features
        soil_input = pd.DataFrame({
            'Temperature': [float(data['temperature'])],
            'Humidity': [float(data['humidity'])],
            'Moisture': [float(data['moisture'])],
            'Soil Type': [le_soil.transform([data['soil_type']])[0]],
            'Nitrogen': [float(data['nitrogen'])],
            'Potassium': [float(data['potassium'])],
            'Phosphorous': [float(data['phosphorous'])],
            'Fertilizer Name': [le_fert.transform([data['fertilizer_name']])[0]]
        })

        # Weather model input
        weather_input = pd.DataFrame({
            'Average_Temperature_C': [float(data['avg_temp'])],
            'Total_Precipitation_mm': [float(data['precipitation'])],
            'CO2_Emissions_MT': [float(data['co2_emissions'])],
            'Extreme_Weather_Events': [int(data['extreme_weather'])],
            'Irrigation_Access_%': [float(data['irrigation_access'])],
            'Pesticide_Use_KG_per_HA': [float(data['pesticide_use'])],
            'Fertilizer_Use_KG_per_HA': [float(data['fertilizer_use'])],
            'Soil_Health_Index': [float(data['soil_health'])],
            'Adaptation_Strategies': [le_strategy.transform([data['adaptation_strategy']])[0]]
        })

        # Get predictions from both models
        soil_pred = soil_model.predict(soil_input)[0]
        soil_prob = soil_model.predict_proba(soil_input)[0]
        soil_crop = le_crop_soil.inverse_transform([soil_pred])[0]
        soil_confidence = float(np.max(soil_prob))

        weather_pred = weather_model.predict(weather_input)[0]
        weather_prob = weather_model.predict_proba(weather_input)[0]
        weather_crop = le_crop_weather.inverse_transform([weather_pred])[0]
        weather_confidence = float(np.max(weather_prob))

        # Combine predictions
        # Strategy 1: Take the prediction with higher confidence
        if soil_confidence > weather_confidence:
            final_prediction = soil_crop
            model_used = "soil (higher confidence)"
            confidence = soil_confidence
        else:
            final_prediction = weather_crop
            model_used = "weather (higher confidence)"
            confidence = weather_confidence

        # If both models predict the same crop, increase confidence
        if soil_crop == weather_crop:
            model_used = "both models agree"
            # Combining confidences - simple average here, but could use other methods
            confidence = (soil_confidence + weather_confidence) / 2

        return jsonify({
            'success': True,
            'prediction': final_prediction,
            'confidence': confidence,
            'model_used': model_used,
            'soil_prediction': {
                'crop': soil_crop,
                'confidence': soil_confidence
            },
            'weather_prediction': {
                'crop': weather_crop,
                'confidence': weather_confidence
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True)