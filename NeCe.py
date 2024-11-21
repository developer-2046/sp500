import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from community import best_partition

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
network_entropies = []
comm_efficiencies = []
dates = []

# Iterate over the data in windows of `window_size`
for start in range(0, len(log_returns) - window_size + 1, window_size):
    end = start + window_size
    window_data = log_returns.iloc[start:end]
    correlation_matrix = window_data.corr()

    # Calculate mean correlation
    mean_correlation = correlation_matrix.mean().mean()
    mean_correlations.append(mean_correlation)

    # Create a network from the correlation matrix using absolute values
    G = nx.Graph()
    for i, symbol_i in enumerate(correlation_matrix.columns):
        for j, symbol_j in enumerate(correlation_matrix.columns):
            if i < j:
                weight = abs(correlation_matrix.iloc[i, j])
                if weight > 0:
                    G.add_edge(symbol_i, symbol_j, weight=weight)

    # Network Entropy
    partition = best_partition(G)
    sizes = np.array(list(dict(pd.Series(partition).value_counts()).values()))
    probs = sizes / sizes.sum()
    entropy = -np.sum(probs * np.log(probs))
    network_entropies.append(entropy)

    # Communication Efficiency
    efficiency = nx.global_efficiency(G)
    comm_efficiencies.append(efficiency)

    last_date = log_returns.index[end - 1]
    dates.append(last_date)

# Create DataFrames for metrics
metrics_df = pd.DataFrame({
    "Mean Correlation": mean_correlations,
    "Network Entropy": network_entropies,
    "Communication Efficiency": comm_efficiencies
}, index=dates)

# Plot the time series of metrics
plt.figure(figsize=(14, 10))

# Plot Mean Correlation
plt.subplot(3, 1, 1)
plt.plot(metrics_df.index, metrics_df["Mean Correlation"], marker='o', linestyle='-')
plt.title("Mean Correlation Over Time")
plt.xlabel("Date")
plt.ylabel("Mean Correlation")
plt.grid(True)

# Plot Network Entropy
plt.subplot(3, 1, 2)
plt.plot(metrics_df.index, metrics_df["Network Entropy"], marker='o', linestyle='-', color='g')
plt.title("Network Entropy Over Time")
plt.xlabel("Date")
plt.ylabel("Network Entropy")
plt.grid(True)

# Plot Communication Efficiency
plt.subplot(3, 1, 3)
plt.plot(metrics_df.index, metrics_df["Communication Efficiency"], marker='o', linestyle='-', color='r')
plt.title("Communication Efficiency Over Time")
plt.xlabel("Date")
plt.ylabel("Communication Efficiency")
plt.grid(True)

plt.tight_layout()
plt.show()
