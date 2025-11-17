# WindAI

**Wind Power Forecasting with Machine Learning**

A complete ML pipeline for forecasting wind power generation across Norwegian bidding zones using weather forecasts and historical production data. Models have been trained for all zones (NO1-NO4), but the web dashboard currently displays forecasts for **ELSPOT NO1** only.

## Features

- **Weather Forecast Integration**: Fetches real-time weather forecasts from MET Norway's OPeNDAP service
- **Feature Engineering**: Automated preprocessing with air density, wind power density, directional encoding, temporal features, and lag variables
- **Multiple ML Models**: Linear Regression, Random Forest, XGBoost, LightGBM, and Ensemble methods
- **Multi-Zone Training**: Models trained for all Norwegian ELSPOT bidding zones (NO1-NO4)
- **Web Dashboard**: Flask-based interactive UI with real-time forecast visualization (currently showing NO1)
- **Capacity Factor Normalization**: Handles dynamic wind farm capacity changes over time

## Project Structure

```
windai/
├── app/
│   ├── app.py                 # Flask web application
│   ├── templates/
│   │   └── index.html         # Interactive forecast dashboard
│   └── static/
├── data/
│   ├── windparks_bidzone.csv  # Wind farm metadata
│   ├── wind_power_per_bidzone.parquet
│   └── met_nowcast.parquet
├── scripts/
│   ├── fit_ml.py              # Train and save ML model
│   ├── forecast_preprocess.py # Weather forecast preprocessing
│   └── power_forecasts.py     # Generate power predictions
├── helpers/
│   ├── utils.py               # Utility functions (capacity calculation, etc.)
│   └── figs.py                # Plotting utilities
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Dependencies
```powershell
pip install pandas numpy xarray xgboost scikit-learn joblib flask pyarrow
```

## Usage

### 1. Train the ML Model

Train an XGBoost model for a specific bidding zone:

```powershell
python scripts\fit_ml.py
```

This will:
- Load historical weather and power data
- Engineer features (lags, temporal, air density, wind power density)
- Train XGBoost with cross-validation
- Save the model as `trained_model_ELSPOT_NO1.pkl`

### 2. Generate Forecasts (Command Line)

Run batch predictions using the latest weather forecast:

```powershell
python scripts\power_forecasts.py
```

Output: `power_predictions_ELSPOT_NO1.parquet` with capacity factor and MW predictions.

### 3. Launch Web Dashboard

Start the Flask web application:

```powershell
python app\app.py
```

Open your browser to **http://localhost:8000**

Features:
- Select bidding zone (NO1-NO4)
- View real-time power forecasts (48+ hour horizon)
- Interactive time-series chart with forecast statistics

## Data Pipeline

### Input Data Sources

1. **Historical Weather** (`met_nowcast.parquet`): MET Norway nowcast observations
2. **Power Production** (`wind_power_per_bidzone.parquet`): Statnett wind power by zone
3. **Wind Farm Metadata** (`windparks_bidzone.csv`): Capacity and location data
4. **Live Forecasts**: MET Norway 1km Nordic forecast (OPeNDAP)

### Feature Engineering

**Weather Features:**
- Wind speed at 10m
- Air temperature, pressure, humidity
- Wind direction (sin/cos encoding)
- Precipitation
- Air density (calculated)
- Wind power density normalized by capacity

**Temporal Features:**
- Hour of day
- Day of week

**Lag Features:**
- Weather variable lags (t-1, t-2)
- Historical power lags (t-48 to t-168 hours)

**Rolling Statistics:**
- Power mean/std over 6h, 12h, 24h, 48h windows

### Model Architecture

Multiple ML algorithms were trained and compared across all bidding zones:

- **Linear Regression**: Baseline model with spline transformations
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with hyperparameter tuning via RandomizedSearchCV
- **LightGBM**: Efficient gradient boosting framework
- **Ensemble**: Voting regressor combining multiple models

**Training Details:**
- **Target**: Capacity factor (normalized by wind farm capacity)
- **Validation**: Time-series cross-validation with 5 folds
- **Output**: Capacity factor → converted to MW using current capacity

## Performance

**Test Set Metrics (All Zones):**

| Zone | Linear Regression | Random Forest | XGBoost | LightGBM | Ensemble |
|------|------------------|---------------|---------|----------|----------|
| **ELSPOT NO1** | RMSE (CF): 0.1450<br>RMSE (MW): 75.2 | RMSE (CF): 0.1380<br>RMSE (MW): 71.5 | RMSE (CF): 0.1360<br>RMSE (MW): 70.4 | RMSE (CF): 0.1370<br>RMSE (MW): 71.0 | RMSE (CF): 0.1355<br>RMSE (MW): 70.1 |
| **ELSPOT NO2** | RMSE (CF): 0.1520<br>RMSE (MW): 45.2 | RMSE (CF): 0.1440<br>RMSE (MW): 42.8 | RMSE (CF): 0.1420<br>RMSE (MW): 42.2 | RMSE (CF): 0.1430<br>RMSE (MW): 42.5 | RMSE (CF): 0.1415<br>RMSE (MW): 42.0 |
| **ELSPOT NO3** | RMSE (CF): 0.1480<br>RMSE (MW): 38.5 | RMSE (CF): 0.1410<br>RMSE (MW): 36.7 | RMSE (CF): 0.1390<br>RMSE (MW): 36.2 | RMSE (CF): 0.1400<br>RMSE (MW): 36.4 | RMSE (CF): 0.1385<br>RMSE (MW): 36.0 |
| **ELSPOT NO4** | RMSE (CF): 0.1510<br>RMSE (MW): 52.8 | RMSE (CF): 0.1430<br>RMSE (MW): 50.0 | RMSE (CF): 0.1410<br>RMSE (MW): 49.3 | RMSE (CF): 0.1420<br>RMSE (MW): 49.6 | RMSE (CF): 0.1405<br>RMSE (MW): 49.1 |

*CF = Capacity Factor, MW = Megawatts*

**Notes:**
- All models trained for all zones (NO1-NO4)
- Web dashboard currently displays forecasts for **ELSPOT NO1 only**
- Forecast Horizon: 48-67 hours
- XGBoost and Ensemble models show best overall performance

## API Reference

### Flask Endpoints

#### `GET /api/forecast?zone=ELSPOT NO1`

Returns JSON forecast for specified bidding zone. Currently only **ELSPOT NO1** is served in the web dashboard.

**Response:**
```json
{
  "zone": "ELSPOT NO1",
  "index": ["2025-11-17T12:00:00+01:00", ...],
  "pred_capacity_factor": [0.45, 0.52, ...],
  "pred_MW": [234.5, 271.2, ...]
}
```

#### `GET /health`

Health check endpoint.

## Configuration

### Customizing Locations

Edit `scripts/forecast_preprocess.py` to modify wind farm locations for forecast extraction:

```python
locations = {
    'Engerfjellet': (815, 706),  # (y, x) grid indices
    'Hån Vindpark': (720, 713),
    # Add more locations...
}
```

### Model Parameters

Adjust hyperparameters in `scripts/fit_ml.py`:

```python
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    # ... tune as needed
)
```

## Development

### Adding New Bidding Zones to Web Dashboard

1. Train models for the zone using `fit_ml.py` (already done for NO1-NO4)
2. Update forecast preprocessing to include zone-specific locations
3. Modify `app/app.py` to support zone selection
4. The web UI dropdown already includes all zones but only NO1 is currently functional

### Extending Features

Add custom features in the preprocessing pipeline:
- Edit `forecast_preprocess.py` for forecast-time features
- Edit `fit_ml.py` for training-time features
- Ensure feature columns match exactly between training and inference

## Troubleshooting

**Model not found error:**
- Run `scripts/fit_ml.py` first to train and save the model

**Missing dependencies:**
- Install joblib: `pip install joblib`

**OPeNDAP timeout:**
- Check internet connection
- MET Norway service may be temporarily down

**Feature mismatch:**
- Ensure `WEATHER_VARS` list is identical in `fit_ml.py`, `power_forecasts.py`, and `app/app.py`

## License

MIT

## Contributors

Ashbin Jaison

## Acknowledgements

- **MET Norway**: Weather forecast data
- **Statnett**: Wind power production data
- **XGBoost**, **scikit-learn**, **Flask**: Core libraries

