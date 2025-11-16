import os
import sys
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Project roots for imports and data paths
APP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
HELPERS_DIR = os.path.join(PROJECT_ROOT, 'helpers')

# Ensure imports resolve from scripts/ and helpers/
sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPTS_DIR)
sys.path.append(HELPERS_DIR)

# Import your preprocessing pipeline and utilities
from forecast_preprocess import avg_wtd_weather_forecast  # noqa: E402
from utils import get_cumulative_capacity  # noqa: E402

app = Flask(__name__, template_folder='templates', static_folder='static')

# Defaults
DEFAULT_ZONE = 'ELSPOT NO1'
MODEL_FILENAME_TMPL = 'trained_model_{zone}.pkl'

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

def build_feature_columns():
    cols = WEATHER_VARS.copy()
    for var in WEATHER_VARS:
        for lag in (1, 2):
            cols.append(f'{var}_lag{lag}')
    cols.extend(['hour', 'day_of_week'])
    return cols

X_COLUMNS = build_feature_columns()

_model_cache = {}

def resolve_model_path(bid_zone: str) -> str:
    fname = MODEL_FILENAME_TMPL.format(zone=bid_zone.replace(' ', '_'))
    # Try project root
    p1 = os.path.join(PROJECT_ROOT, fname)
    if os.path.exists(p1):
        return p1
    # Try models/ folder if you later move models there
    p2 = os.path.join(PROJECT_ROOT, 'models', fname)
    if os.path.exists(p2):
        return p2
    return p1  # default to project root path

def load_model(bid_zone: str):
    if bid_zone in _model_cache:
        return _model_cache[bid_zone]
    model_path = resolve_model_path(bid_zone)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for zone {bid_zone}: {model_path}")
    model = joblib.load(model_path)
    _model_cache[bid_zone] = model
    return model

def load_capacity_df():
    windparks_csv = os.path.join(PROJECT_ROOT, 'data', 'windparks_bidzone.csv')
    windparks = pd.read_csv(windparks_csv)
    capacity_df = get_cumulative_capacity(windparks)
    # Make tz-aware to align with forecasts (avg_wtd_weather_forecast returns CET)
    if capacity_df.index.tz is None:
        capacity_df = capacity_df.tz_localize('UTC').tz_convert('CET')
    else:
        # Ensure CET
        capacity_df = capacity_df.tz_convert('CET')
    return capacity_df

def predict_for_zone(bid_zone: str):
    # 1) Get engineered forecast features
    features_df = avg_wtd_weather_forecast()
    if features_df is None or features_df.empty:
        raise ValueError('Preprocessed forecast features are empty')
    # Remove any NaNs due to lags
    features_df = features_df.dropna()

    # 2) Load model
    model = load_model(bid_zone)

    # 3) Align features
    missing = [c for c in X_COLUMNS if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    X = features_df[X_COLUMNS]

    # 4) Predict capacity factor
    cf_pred = model.predict(X)

    # 5) Convert to MW using capacity
    capacity_df = load_capacity_df()
    cap_series = capacity_df[bid_zone].reindex(X.index, method='ffill')
    results = pd.DataFrame(index=X.index)
    results['pred_capacity_factor'] = cf_pred
    results['pred_MW'] = (results['pred_capacity_factor'].clip(0, 1)) * cap_series

    # Add a few context columns for UI
    for c in ('wind_speed_10m', 'air_temperature_2m', 'air_pressure_at_sea_level'):
        if c in features_df.columns:
            results[c] = features_df[c]

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/forecast', methods=['GET'])
def api_forecast():
    bid_zone = request.args.get('zone', DEFAULT_ZONE)
    try:
        df = predict_for_zone(bid_zone)
        payload = {
            'zone': bid_zone,
            'index': [ts.isoformat() for ts in df.index.to_pydatetime()],
            'pred_capacity_factor': df['pred_capacity_factor'].tolist(),
            'pred_MW': [None if pd.isna(x) else float(x) for x in df['pred_MW'].tolist()],
        }
        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
