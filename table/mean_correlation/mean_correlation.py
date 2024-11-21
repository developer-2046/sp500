import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('../../SP500_25May24_cleaned_filled.csv', index_col="Date", parse_dates=True)

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

# Plot the time series of mean correlations
plt.figure(figsize=(14, 7))
plt.plot(mean_correlation_df.index, mean_correlation_df["Mean Correlation"], marker='o', linestyle='-')
plt.title("Mean Correlation Over Time")
plt.xlabel("Date")
plt.ylabel("Mean Correlation")
plt.grid(True)
plt.show()
