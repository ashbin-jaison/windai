import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import pandas as pd


def open_opendap_dataset(url):
    try:
        ds = xr.open_dataset(url)
        return ds
    except Exception as e:
        print(f"Error opening OPeNDAP dataset: {e}")
        return None
    
def load_data():
    url = "https://thredds.met.no/thredds/dodsC/metpplatest/met_forecast_1_0km_nordic_latest.nc"
    ds = open_opendap_dataset(url)
    return ds


def get_forecast_dataframe(ds, lat, lon, operating_power_max, variables=None):
    """
    Extract forecast data and return as a pandas DataFrame.
    
    Args:
        ds: xarray Dataset
        lat: Latitude (degrees)
        lon: Longitude (degrees)
        variables: List of variable names to extract (None = all)
    
    Returns:
        pandas DataFrame with time index and forecast variables as columns
    """
    point_ds = ds[variables].isel(y=lat, x=lon)
    df = point_ds.to_dataframe()

    # 1. Calculate air density
    R = 287.05  # Gas constant for dry air in J/(kg·K)
    df['air_density'] = (df['air_pressure_at_sea_level'] * 100) / (R * df['air_temperature_2m'])

   # 2. Calculate wind power density normalized by capacity
    df['wind_power_density_normalized'] = (
        0.5 * df['air_density'] * df['wind_speed_10m']**3
    ) / operating_power_max
    
    # 3. Convert wind direction to sin/cos
    wind_dir_rad = np.radians(df['wind_direction_10m'])
    df['wind_direction_sin'] = np.sin(wind_dir_rad)
    df['wind_direction_cos'] = np.cos(wind_dir_rad)

    return df

def get_forecast_multiple_locations(ds, locations, operating_power_max, variables=None):
    """
    Extract forecast data for multiple lat/lon locations.
    
    Args:
        ds: xarray Dataset
        locations:  Dict with location names {'location1': (lat1, lon1), ...}
        variables: List of variable names to extract (None = all)
    
    Returns:
        Dictionary of DataFrames {location_name: forecast_df}
    """
    results = {}
    
    for name, (lat, lon) in locations.items():
        df = get_forecast_dataframe(ds, lat, lon, operating_power_max[name], variables=variables)
        results[name] = df
    return results
    
 
def get_weighted_average_forecast(ds, locations, weights, operating_power_max, variables=None):
    """
    Get weighted average forecast across multiple locations.
    Useful for averaging forecasts across a wind farm area.
    
    Args:
        ds: xarray Dataset
        locations: List of tuples [(lat1, lon1), (lat2, lon2), ...] or 
                   Dict with location names
        weights: List or dict of weights for each location (None = equal weights)
                 Weights will be normalized to sum to 1
        variables: List of variable names to extract
    
    Returns:
        pandas DataFrame with weighted average forecast
    """
    forecasts = get_forecast_multiple_locations(ds, locations, operating_power_max, variables)
    
    # Convert to list of DataFrames
    df_list = list(forecasts.values())
    
    # Set up weights
    if weights is None:
        weights = [1.0 / len(df_list)] * len(df_list)
    elif isinstance(weights, dict):
        weights = list(weights.values())
    
    # Normalize weights to sum to 1
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Weighted average
    weighted_df = sum(df * w for df, w in zip(df_list, weights))
    
    return weighted_df

def avg_wtd_weather_forecast():
    ds = load_data()
    if ds:
        print("Dataset loaded successfully. Summary:")
        print("\n" + "="*50)
        
        # Example 1: Extract forecast for multiple named locations
        locations = {
            'Engerfjellet': (815, 706),
            'Hån Vindpark': (720, 713),
            'Kjølberget': (887, 747),
            'Marker Vindpark': (715, 714),
            'Raskiftet': (908, 727),
            'Songkjølen': (815, 706)
        }
        
        variables = ['wind_speed_10m', 'air_temperature_2m', 'air_pressure_at_sea_level','relative_humidity_2m','precipitation_amount','wind_direction_10m']
        weights = {'Engerfjellet': 52.8, 'Hån Vindpark': 21, 'Kjølberget': 55.9,'Marker Vindpark': 54, 'Raskiftet': 111.6, 'Songkjølen': 110.4}

        
        # Weighted average across locations 
        print("\n" + "="*50)
        print("\nWeighted average forecast:")
        avg_forecast = get_weighted_average_forecast(ds, locations, weights, operating_power_max=weights, variables=variables)
        avg_forecast = avg_forecast.tz_localize('UTC').tz_convert('CET')

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
                avg_forecast[f'{var}_lag{lag}'] = avg_forecast[var].shift(lag)

        # Add temporal features
        avg_forecast['hour'] = avg_forecast.index.hour
        avg_forecast['day_of_week'] = avg_forecast.index.dayofweek

        #print(avg_forecast.head(10))

        return avg_forecast