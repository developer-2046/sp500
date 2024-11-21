import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# Load the data
df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date", parse_dates=True)

# Filter the dataset for the date range from 2018 to 2022
df = df.loc['2018-01-01':'2022-12-31']

# Parse the sector information from the column names
sectors = {col.split('_')[0]: col.split('_')[1] for col in df.columns}
df.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], sectors[col.split('_')[0]]) for col in df.columns], names=['Symbol', 'Sector'])

# Sort columns by sector
df = df.sort_index(axis=1, level=1)

# Calculate log returns
log_returns = np.log(df / df.shift(1))

# Number of days per window
window_size = 20

mean_correlations = []
dates = []

# Iterate over the data in windows of `window_size`
for start in range(0, len(log_returns) - window_size + 1, window_size):
    end = start + window_size
    window_data = log_returns.iloc[start:end]
    correlation_matrix = window_data.corr()
    mean_correlation = correlation_matrix.mean().mean()
    mean_correlations.append(mean_correlation)
    last_date = log_returns.index[end - 1]  # Get the last date of the current window
    dates.append(last_date)

# Create a DataFrame for mean correlations
mean_correlation_df = pd.DataFrame(mean_correlations, index=dates, columns=["Mean Correlation"])

# Identify peaks and valleys in the mean correlations
peaks, _ = find_peaks(mean_correlation_df["Mean Correlation"])
valleys, _ = find_peaks(-mean_correlation_df["Mean Correlation"])

# Sort the peaks and valleys based on their values
sorted_peaks = np.argsort(mean_correlation_df["Mean Correlation"].iloc[peaks])[-3:]  # Highest 3 peaks
sorted_valleys = np.argsort(mean_correlation_df["Mean Correlation"].iloc[valleys])[:3]  # Lowest 3 valleys

# Identify the two middle points
middle_index = len(mean_correlation_df) // 2
middle_points = [middle_index - 1, middle_index + 1]  # Select two nearby middle points

# Combine the indexes of peaks, valleys, and middle points
selected_indices = np.concatenate([peaks[sorted_peaks], valleys[sorted_valleys], middle_points])
selected_indices = np.sort(selected_indices)  # Sort the combined indices in ascending order

# Plot the selected points and connect them
plt.figure(figsize=(12, 6))

# Plot the line connecting the selected points
plt.plot(mean_correlation_df.index[selected_indices], mean_correlation_df["Mean Correlation"].iloc[selected_indices], linestyle='-', marker='o')

# Mark each point with different colors and labels
for idx in selected_indices:
    date = mean_correlation_df.index[idx]
    value = mean_correlation_df["Mean Correlation"].iloc[idx]

    if idx in peaks[sorted_peaks]:
        plt.plot(date, value, 'rx', markersize=10, label=f'Peak on {date.strftime("%Y-%m-%d")}')
    elif idx in valleys[sorted_valleys]:
        plt.plot(date, value, 'bx', markersize=10, label=f'Valley on {date.strftime("%Y-%m-%d")}')
    elif idx in middle_points:
        plt.plot(date, value, 'go', markersize=10, label=f'Middle on {date.strftime("%Y-%m-%d")}')

# Customize the plot
plt.title("Top 3 Peaks, Top 3 Valleys, and 2 Middle Points Connected")
plt.xlabel("Date")
plt.ylabel("Mean Correlation")
plt.grid(True)
plt.legend()

# Save and show the plot
output_dir = "selected_points_plot"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "peaks_valleys_middle_connected.png"))
plt.show()
