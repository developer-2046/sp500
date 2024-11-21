import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Load the data
df = pd.read_csv('SP500_25May24_cleaned_filled.csv', index_col="Date", parse_dates=True)

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
average_degrees = []
dates = []

# Iterate over the data in windows of `window_size`
for start in range(0, len(log_returns) - window_size + 1, window_size):
    end = start + window_size
    window_data = log_returns.iloc[start:end]
    correlation_matrix = window_data.corr()
    mean_correlation = correlation_matrix.mean().mean()

    # Create a graph from the correlation matrix
    G = nx.from_pandas_adjacency(correlation_matrix)
    avg_degree = np.mean([degree for node, degree in G.degree()])

    mean_correlations.append(mean_correlation)
    average_degrees.append(avg_degree)

    last_date = log_returns.index[end - 1]  # Get the last date of the current window
    dates.append(last_date)

# Create DataFrames for mean correlations and average degrees
mean_correlation_df = pd.DataFrame(mean_correlations, index=dates, columns=["Mean Correlation"])
average_degree_df = pd.DataFrame(average_degrees, index=dates, columns=["Average Degree"])

# Plot Mean Correlation and Average Degree over time
fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Mean Correlation', color=color)
ax1.plot(mean_correlation_df.index, mean_correlation_df["Mean Correlation"], marker='o', linestyle='-', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Average Degree', color=color)
ax2.plot(average_degree_df.index, average_degree_df["Average Degree"], marker='x', linestyle='--', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Mean Correlation and Average Degree Over Time")
plt.grid(True)
fig.tight_layout()
plt.show()
