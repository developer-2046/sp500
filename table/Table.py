import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

# Create a market index by averaging the closing prices of all stocks
market_index = df.mean(axis=1)

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

# Calculate the 75th and 25th percentile thresholds for the market index
high_threshold = market_index.quantile(0.75)
low_threshold = market_index.quantile(0.25)

# Plot a boxplot of the market index values
plt.figure(figsize=(10, 5))
plt.boxplot(market_index)
plt.axhline(y=high_threshold, color='r', linestyle='--', label='75th Percentile')
plt.axhline(y=low_threshold, color='b', linestyle='--', label='25th Percentile')
plt.title('Market Index Boxplot with 25th and 75th Percentile Thresholds')
plt.ylabel('Market Index Value')
plt.legend()
plt.show()

# Define a function to determine if the market was high, low, or normal
def get_market_status(date, market_index, high_threshold, low_threshold):
    market_value = market_index.loc[date]
    if market_value > high_threshold:
        return 'High'
    elif market_value < low_threshold:
        return 'Low'
    else:
        return 'Normal'

# Generate the comparison table
comparison_table = []

for peak in peaks:
    date = mean_correlation_df.index[peak]
    market_status = get_market_status(date, market_index, high_threshold, low_threshold)
    comparison_table.append((date.strftime('%Y-%m-%d'), 'Peak', market_status))

for valley in valleys:
    date = mean_correlation_df.index[valley]
    market_status = get_market_status(date, market_index, high_threshold, low_threshold)
    comparison_table.append((date.strftime('%Y-%m-%d'), 'Valley', market_status))

# Convert the comparison table to a DataFrame
comparison_df = pd.DataFrame(comparison_table, columns=['Date', 'Mean Correlation Result', 'Actual Market Result'])

# Display the comparison table
print(comparison_df)

# Plot the time series of mean correlations with peaks and valleys marked
plt.figure(figsize=(14, 7))
plt.plot(mean_correlation_df.index, mean_correlation_df["Mean Correlation"], marker='o', linestyle='-')
plt.plot(mean_correlation_df.index[peaks], mean_correlation_df["Mean Correlation"].iloc[peaks], "x", label="Peaks", color='red')
plt.plot(mean_correlation_df.index[valleys], mean_correlation_df["Mean Correlation"].iloc[valleys], "x", label="Valleys", color='blue')
plt.title("Mean Correlation Over Time (2018-2022)")
plt.xlabel("Date")
plt.ylabel("Mean Correlation")
plt.grid(True)
plt.legend()
plt.show()

