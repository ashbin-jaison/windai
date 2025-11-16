import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats

from utils import get_cumulative_capacity

# Read and preprocess data
weather_nowcast = pd.read_parquet('data/met_nowcast.parquet')
weather_nowcast = weather_nowcast.tz_localize('UTC').tz_convert('CET')

windpower = pd.read_parquet('data/wind_power_per_bidzone.parquet')
windpower = windpower.tz_localize('UTC').tz_convert('CET')

windparks = pd.read_csv('data/windparks_bidzone.csv')
capacity_df = get_cumulative_capacity(windparks)
capacity_df = capacity_df.tz_localize('UTC').tz_convert('CET')
capacity_df = capacity_df.reindex(windpower.index, method='ffill')

# Calculate capacity factors
for zone in windpower.columns:
    windpower[str(zone)] = windpower[str(zone)]/capacity_df[str(zone)]

# Select bidzone
bid_zone = 'ELSPOT NO1'
_windparks_in_bid_zone = windparks[windparks['bidding_area'] == bid_zone]
_windpower_in_bid_zone = windpower[bid_zone]

# Process weather data with weights
if weather_nowcast.index.name == 'time':
    weather_nowcast = weather_nowcast.reset_index()

_weather_with_parks = weather_nowcast.merge(
    _windparks_in_bid_zone[['substation_name', 'operating_power_max']], 
    left_on='windpark', 
    right_on='substation_name',
    how='inner'
)

# Calculate physics-based features
def calculate_air_density(temp_K, pressure_Pa):
    R = 287.05  # Gas constant for dry air in J/(kg·K)
    return pressure_Pa / (R * temp_K)

_weather_with_parks['air_density'] = calculate_air_density(
    _weather_with_parks['air_temperature_2m'],
    _weather_with_parks['air_pressure_at_sea_level'] * 100
)

_weather_with_parks['wind_power_density_normalized'] = (
    0.5 * 
    _weather_with_parks['air_density'] * 
    _weather_with_parks['wind_speed_10m']**3
) / _weather_with_parks['operating_power_max']

wind_dir_rad = np.radians(_weather_with_parks['wind_direction_10m'])
_weather_with_parks['wind_direction_sin'] = np.sin(wind_dir_rad)
_weather_with_parks['wind_direction_cos'] = np.cos(wind_dir_rad)

# Calculate weighted averages
_weather_nowcast_in_bid_zone = _weather_with_parks.groupby('time').apply(
    lambda x: pd.Series({
        col: np.average(x[col], weights=x['operating_power_max']) 
        for col in x.columns 
        if col not in ['time', 'windpark', 'substation_name', 'operating_power_max']
    })
)

# Prepare final dataset
data_bidzone = pd.concat([_windpower_in_bid_zone, _weather_nowcast_in_bid_zone], axis=1)
data_bidzone = data_bidzone.loc[_windparks_in_bid_zone['prod_start_new'].max():].dropna()

def plot_lag_analysis(data, var_name, max_lags=24):
    """
    Analyze and plot lag relationships for a given variable.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the variable and power output
    var_name : str
        Name of the variable to analyze
    max_lags : int
        Maximum number of lags to consider
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: ACF
    plt.subplot(2, 2, 1)
    acf_values = acf(data[var_name], nlags=max_lags)
    plt.plot(range(max_lags + 1), acf_values)
    plt.fill_between(range(max_lags + 1), 
                     -1.96/np.sqrt(len(data)), 
                     1.96/np.sqrt(len(data)), 
                     alpha=0.2)
    plt.title(f'ACF for {var_name}')
    plt.xlabel('Lag')
    plt.grid(True)
    
    # Plot 2: PACF
    plt.subplot(2, 2, 2)
    pacf_values = pacf(data[var_name], nlags=max_lags)
    plt.plot(range(max_lags + 1), pacf_values)
    plt.fill_between(range(max_lags + 1), 
                     -1.96/np.sqrt(len(data)), 
                     1.96/np.sqrt(len(data)), 
                     alpha=0.2)
    plt.title(f'PACF for {var_name}')
    plt.xlabel('Lag')
    plt.grid(True)
    
    # Plot 3: Cross-correlation with power output
    plt.subplot(2, 2, 3)
    cross_corr = [data[var_name].shift(i).corr(data[bid_zone]) for i in range(max_lags + 1)]
    plt.plot(range(max_lags + 1), cross_corr)
    plt.title(f'Cross-correlation with Power Output')
    plt.xlabel('Lag')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print significant lags
    significant_lags = []
    for lag in range(1, max_lags + 1):
        corr = data[var_name].shift(lag).corr(data[bid_zone])
        t_stat = corr * np.sqrt((len(data)-2)/(1-corr**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(data)-2))
        if p_value < 0.05:
            significant_lags.append((lag, corr, p_value))
    
    print(f"\nSignificant lags for {var_name}:")
    for lag, corr, p_value in significant_lags:
        print(f"Lag {lag}: correlation = {corr:.3f}, p-value = {p_value:.3e}")
    
    return significant_lags

if __name__ == "__main__":
    # Analyze key variables
    variables_to_analyze = [
        'wind_speed_10m', 
        'wind_power_density_normalized',
        'air_temperature_2m',
        'air_pressure_at_sea_level'
    ]
    
    results = {}
    for var in variables_to_analyze:
        print(f"\nAnalyzing {var}...")
        results[var] = plot_lag_analysis(data_bidzone, var)
        
    # Save results to file
    with open('lag_analysis_results.txt', 'w') as f:
        f.write(f"Lag Analysis Results for {bid_zone}\n")
        f.write("=" * 50 + "\n\n")
        for var, lags in results.items():
            f.write(f"\n{var}:\n")
            f.write("-" * 30 + "\n")
            for lag, corr, p_value in lags:
                f.write(f"Lag {lag}: correlation = {corr:.3f}, p-value = {p_value:.3e}\n")