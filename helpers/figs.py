import pandas as pd
import matplotlib.pyplot as plt

# Statnett Windpark Metadata
windparks = pd.read_csv('data/windparks_bidzone.csv')
print(windparks.head())

# Convert prod_start_new to datetime
windparks['prod_start_new'] = pd.to_datetime(windparks['prod_start_new'])

# Create time series plot for each bidding area
plt.figure(figsize=(15, 8))
for area in windparks['bidding_area'].unique():
    area_data = windparks[windparks['bidding_area'] == area]
    # Sort by date and plot cumulative capacity
    area_data = area_data.sort_values('prod_start_new')
    plt.plot(area_data['prod_start_new'], 
             area_data['operating_power_max'].cumsum(),
             label=area,
             marker='o')

plt.title('Wind Power Capacity Growth by Bidding Area')
plt.xlabel('Time')
plt.ylabel('Cumulative Power Capacity (MW)')
plt.legend(title='Bidding Area')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nFinal capacity per bidding area (MW):")
final_capacity = windparks.groupby('bidding_area')['operating_power_max'].sum()
print(final_capacity)