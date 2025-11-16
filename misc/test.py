import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# You will need pyarrow for reading the parquet files
import pandas as pd
import numpy as np
# Scikitlearn is used for training an example model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
import matplotlib.pyplot as plt

from helpers.utils import get_cumulative_capacity, plot_cumulative_capacity

#MET nowcast
weather_nowcast = pd.read_parquet('data/met_nowcast.parquet')
weather_nowcast = weather_nowcast.tz_localize('UTC').tz_convert('CET') # Converting to local time

# Statnett Windpower Data
windpower = pd.read_parquet('data/wind_power_per_bidzone.parquet')
windpower = windpower.tz_localize('UTC').tz_convert('CET')

# Statnett Windpower Data
windparks = pd.read_csv('data/windparks_bidzone.csv')
capacity_df = get_cumulative_capacity(windparks)

# Convert capacity_df to the same timezone as windpower
capacity_df = capacity_df.tz_localize('UTC').tz_convert('CET')
# Reindex capacity_df to match windpower timestamps
capacity_df = capacity_df.reindex(windpower.index, method='ffill')
#plot_cumulative_capacity(capacity_df)

# Getting the capacity factor estimates
for zone in windpower.columns:
    windpower[str(zone)] = windpower[str(zone)]/capacity_df[str(zone)]

result = []
# Modelling this bid zone
bid_zone = 'ELSPOT NO1'

for bid_zone in ['ELSPOT NO1', 'ELSPOT NO2', 'ELSPOT NO3', 'ELSPOT NO4']:  # You can add more bid zones here

    # Selecting the windparks in bid zone from metadata
    _windparks_in_bid_zone = windparks[windparks['bidding_area'] == bid_zone]

    # Selecting the windpower from bid zone
    _windpower_in_bid_zone = windpower[bid_zone]

    # Calculate weighted average weather using operating_power_max as weights

    # Reset index if time is in the index
    if weather_nowcast.index.name == 'time':
        weather_nowcast = weather_nowcast.reset_index()

    # First merge weather data with windpark weights
    _weather_with_parks = weather_nowcast.merge(
        _windparks_in_bid_zone[['substation_name', 'operating_power_max']], 
        left_on='windpark', 
        right_on='substation_name',
        how='inner'
    )


    # Calculate air density and wind power for each record
    def calculate_air_density(temp_K, pressure_Pa):
        R = 287.05  # Gas constant for dry air in J/(kg·K)
        return pressure_Pa / (R * temp_K)

    _weather_with_parks['air_density'] = calculate_air_density(
        _weather_with_parks['air_temperature_2m'],
        _weather_with_parks['air_pressure_at_sea_level'] * 100  # Convert hPa to Pa
    )

    # Calculate wind power density normalized by capacity
    _weather_with_parks['wind_power_density_normalized'] = (
        0.5 * 
        _weather_with_parks['air_density'] * 
        _weather_with_parks['wind_speed_10m']**3
    ) / _weather_with_parks['operating_power_max']

    # Convert wind direction to radians and calculate sin/cos
    wind_dir_rad = np.radians(_weather_with_parks['wind_direction_10m'])
    _weather_with_parks['wind_direction_sin'] = np.sin(wind_dir_rad)
    _weather_with_parks['wind_direction_cos'] = np.cos(wind_dir_rad)

    # Group by time and calculate weighted average
    _weather_nowcast_in_bid_zone = _weather_with_parks.groupby('time').apply(
        lambda x: pd.Series({
            col: np.average(x[col], weights=x['operating_power_max']) 
            for col in x.columns 
            if col not in ['time', 'windpark', 'substation_name', 'operating_power_max']
        })
    )




    # Concatenating datasets (weather and power) into one dataframe
    data_bidzone = pd.concat([_windpower_in_bid_zone, _weather_nowcast_in_bid_zone], axis=1)

    # Filtering out data where not all windparks are operational
    data_bidzone = data_bidzone.loc[_windparks_in_bid_zone['prod_start_new'].max():].dropna()

    # Add lagged features
    weather_vars = [
        'wind_speed_10m', 
        'air_pressure_at_sea_level', 
        'air_temperature_2m', 
        'relative_humidity_2m',
        'wind_power_density_normalized',
        'wind_direction_sin',
        'wind_direction_cos',
        'precipitation_amount'
    ]

    # Create lags
    for var in weather_vars:
        for lag in range(1, 3):  # Create 2 lags
            data_bidzone[f'{var}_lag{lag}'] = data_bidzone[var].shift(lag)

    # Add temporal features
    data_bidzone['hour'] = data_bidzone.index.hour
    data_bidzone['day_of_week'] = data_bidzone.index.dayofweek

    # Remove rows with NaN from lagged values
    data_bidzone = data_bidzone.dropna()

    # Train/test split
    df_train = data_bidzone.loc[:'2025-01-01']
    df_test = data_bidzone.loc['2025-01-01':].copy()

    # Base features plus lagged features
    x_columns = weather_vars.copy()  # Start with current values
    for var in weather_vars:
        for lag in range(1, 3):
            x_columns.append(f'{var}_lag{lag}')

    # Add temporal features
    x_columns.extend(['hour', 'day_of_week'])

    y_column = bid_zone

    #print(x_columns)
    if bid_zone == 'ELSPOT NO3':
        # Creating a linear regression model with spline transformer on the input data.
        model = make_pipeline(SplineTransformer(n_knots=5, degree=3, include_bias=True), LinearRegression(fit_intercept=True), )
        model.fit(df_train[x_columns], df_train[y_column])
        
    if bid_zone == 'ELSPOT NO2' or bid_zone=='ELSPOT NO4':
        # Random Forest Regressor
        model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, 
                                    min_samples_leaf=2, random_state=42, n_jobs=-1)
        model.fit(df_train[x_columns], df_train[y_column])

    if bid_zone == 'ELSPOT NO1':
        # XGBoost Regressor
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, 
                            min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1)
        model.fit(df_train[x_columns], df_train[y_column])


    # LightGBM Regressor
    #model = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, 
    #                      max_depth=6, min_child_samples=20, subsample=0.8, 
    #                      colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    #model.fit(df_train[x_columns], df_train[y_column])

    # Ensemble Model - Voting Regressor
    #model = VotingRegressor([
    # ('lgb', LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, 
    #                      max_depth=6, min_child_samples=20, subsample=0.8, 
    #                      colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)),
    # ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, 
    #                     min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
    #                     random_state=42, n_jobs=-1)),
    # ('rf', RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, 
    #                             min_samples_leaf=2, random_state=42, n_jobs=-1))
    #])
    #model.fit(df_train[x_columns], df_train[y_column])

    df_test.loc[:, 'Predicted'] = model.predict(df_test[x_columns])


    df_test.loc[:, 'modelled'] = model.predict(df_test[x_columns])
    # Convert normalized values (capacity factor) back to MW using capacity
    capacity_series = capacity_df[bid_zone].reindex(df_test.index, method='ffill')

    # Denormalize both observed and modelled to MW
    df_test.loc[:, 'observed_MW'] = df_test[bid_zone] * capacity_series
    df_test.loc[:, 'modelled_MW'] = df_test['modelled'] * capacity_series

    # Report errors in both normalized units and MW
    rmse_cf = root_mean_squared_error(df_test[bid_zone], df_test['modelled'])
    rmse_mw = root_mean_squared_error(df_test['observed_MW'], df_test['modelled_MW'])
    print(f'Test RMSE (capacity factor): {rmse_cf:.4f}')
    print(f'Test RMSE (MW): {rmse_mw:.2f} MW')

    result.append(df_test[["observed_MW", "modelled_MW"]])
    del data_bidzone

final_df = pd.concat(result, ignore_index=True)
print(final_df.head(5))

col1 = final_df.iloc[:, 0]
col2 = final_df.iloc[:, 1]

rmse = np.sqrt(((col1 - col2)**2).mean())
print(f'Overall RMSE (MW): {rmse:.2f} MW')