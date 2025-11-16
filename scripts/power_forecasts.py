import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
from forecast_preprocess import avg_wtd_weather_forecast
from helpers.utils import get_cumulative_capacity

# Configuration
BID_ZONE = 'ELSPOT NO1'
MODEL_PATH = f'trained_model_{BID_ZONE.replace(" ", "_")}.pkl'

# Feature columns (must match training exactly)
WEATHER_VARS = [
    'wind_speed_10m', 
    'air_pressure_at_sea_level', 
    'air_temperature_2m', 
    'relative_humidity_2m',
    'wind_power_density_normalized',
    'wind_direction_sin',
    'wind_direction_cos',
    'precipitation_amount'
]

# Build feature columns: base vars + lags + temporal
x_columns = WEATHER_VARS.copy()
for var in WEATHER_VARS:
    for lag in range(1, 3):
        x_columns.append(f'{var}_lag{lag}')
x_columns.extend(['hour', 'day_of_week'])

# Load preprocessed forecast with all feature
avg_forecast = avg_wtd_weather_forecast()

# Remove NaN rows from lags
avg_forecast_clean = avg_forecast.dropna()

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run fit_ml.py first to train and save the model.")

model = joblib.load(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

# Predict capacity factor
predictions_cf = model.predict(avg_forecast_clean[x_columns])

# Load capacity data for MW conversion
windparks = pd.read_csv('data/windparks_bidzone.csv')
capacity_df = get_cumulative_capacity(windparks)
capacity_df = capacity_df.tz_localize('UTC').tz_convert('CET')

# Get capacity for prediction timestamps
capacity_series = capacity_df[BID_ZONE].reindex(avg_forecast_clean.index, method='ffill')

# Convert to MW
predictions_mw = predictions_cf * capacity_series

# Build results DataFrame
results = pd.DataFrame(index=avg_forecast_clean.index)
results['predicted_capacity_factor'] = predictions_cf
results['predicted_MW'] = predictions_mw

# Add key weather variables for reference
results['wind_speed_10m'] = avg_forecast_clean['wind_speed_10m']
results['air_temperature_2m'] = avg_forecast_clean['air_temperature_2m']
results['wind_power_density_normalized'] = avg_forecast_clean['wind_power_density_normalized']

print("\n" + "="*50)
print("Power Forecast Results:")
print("="*50)
print(f"\nForecast horizon: {len(results)} hours")
print(f"From: {results.index[0]}")
print(f"To:   {results.index[-1]}")
print(f"\nPredicted power range: {results['predicted_MW'].min():.2f} - {results['predicted_MW'].max():.2f} MW")
print(f"Mean predicted power: {results['predicted_MW'].mean():.2f} MW")
print("\nFirst 12 hours:")
print(results.head(12))

# Save predictions
output_file = f'power_predictions_{BID_ZONE.replace(" ", "_")}.parquet'
results.to_parquet(output_file)
print(f"\nSaved predictions to: {output_file}")
