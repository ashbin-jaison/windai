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

##### One time function to get and y and x from metcoop
def get_forecast_at_location(ds, lat, lon, variables=None):
    """
    Extract forecast data for a specific latitude and longitude.
    Works with curvilinear grids where lat/lon are 2D coordinate arrays.
    
    Args:
        ds: xarray Dataset
        lat: Latitude (degrees)
        lon: Longitude (degrees)
        variables: List of variable names to extract (None = all)
    
    Returns:
        xarray Dataset with forecast data at the specified location
    """
    # For curvilinear grids, lat/lon are 2D arrays indexed by (y, x)
    # Find the nearest grid point by computing distance
    
    # Get lat/lon coordinates
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    
    # Calculate distance from target point to all grid points
    # Using simple Euclidean distance (for more accuracy, use Haversine)
    dist = np.sqrt((lats - lat)**2 + (lons - lon)**2)
    
    # Find indices of minimum distance
    y_idx, x_idx = np.unravel_index(dist.argmin(), dist.shape)

    print(y_idx, x_idx)
    
    # Select data at this grid point
    point_ds = ds.isel(y=y_idx, x=x_idx)
    
    # Optionally filter variables
    if variables:
        point_ds = point_ds[variables]
    
    return point_ds

    # Example 1: Extract forecast for multiple named locations
    locations = {
        'Engerfjellet': (60.3566, 11.5185),
        'Hån Vindpark': (59.5084, 11.7401),
        'Kjølberget': (61.0178, 12.2126),
        'Marker Vindpark': (59.4592, 11.7497),
        'Raskiftet': (61.1927, 11.8201),
        'Songkjølen': (60.3566, 11.5185)
    }

ds = load_data()
# Select data at this grid point
point_ds = ds[['air_temperature_2m', 'precipitation_amount', 'wind_speed_10m']].isel(y=815, x=706)
#print(point_ds['air_temperature_2m'].values)
df = point_ds.to_dataframe()
print(df.head())