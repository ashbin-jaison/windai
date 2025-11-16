import pandas as pd
import matplotlib.pyplot as plt

def get_cumulative_capacity(windparks_df):
    """
    Create a DataFrame with cumulative wind power capacity over time per bidding area.
    
    Args:
        windparks_df (pd.DataFrame): DataFrame containing windpark data with columns 
                                   'prod_start_new', 'operating_power_max', and 'bidding_area'
    
    Returns:
        pd.DataFrame: DataFrame with time as index and bidding areas as columns,
                     containing cumulative capacity values
    """
    # Convert prod_start_new to datetime if not already
    windparks_df = windparks_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(windparks_df['prod_start_new']):
        windparks_df['prod_start_new'] = pd.to_datetime(windparks_df['prod_start_new'])
    
    # Create pivot table with time and bidding areas
    capacity_df = windparks_df.pivot_table(
        values='operating_power_max',
        index='prod_start_new',
        columns='bidding_area',
        aggfunc='sum'
    ).sort_index()
    
    # Forward fill to get cumulative capacity
    capacity_df = capacity_df.cumsum().ffill()
    
    return capacity_df

def plot_cumulative_capacity(capacity_df, title='Cumulative Wind Power Capacity by Bidding Area'):
    """
    Plot cumulative capacity over time for each bidding area.
    
    Args:
        capacity_df (pd.DataFrame): DataFrame with time as index and bidding areas as columns
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    for column in capacity_df.columns:
        plt.plot(capacity_df.index, capacity_df[column], label=column, marker='o')
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Capacity (MW)')
    plt.legend(title='Bidding Area')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()