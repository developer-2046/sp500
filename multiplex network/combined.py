import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from community import best_partition
from scipy.stats import entropy

# Load the data
df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date", parse_dates=True)

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
multilayer_network_entropies = []
shannon_entropies = []
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
                weight = abs(correlation_matrix.iloc[i, j])  # Use absolute value
                if weight > 0:
                    G.add_edge(symbol_i, symbol_j, weight=weight)

    # Network Entropy
    degrees = np.array([deg for _, deg in G.degree()])
    unique, counts = np.unique(degrees, return_counts=True)
    degree_probabilities = counts / counts.sum()
    network_entropy = entropy(degree_probabilities)
    network_entropies.append(network_entropy)

    # Multilayer Network Entropy
    partition = best_partition(G, weight='weight')
    sizes = np.array(list(dict(pd.Series(partition).value_counts()).values()))
    probs = sizes / sizes.sum()
    multilayer_network_entropy = -np.sum(probs * np.log(probs))
    multilayer_network_entropies.append(multilayer_network_entropy)

    # Shannon Entropy
    corr_values = correlation_matrix.values.flatten()
    corr_values = corr_values[corr_values != 1]  # Exclude self-correlation
    corr_probabilities = np.abs(corr_values) / np.sum(np.abs(corr_values))
    shannon_entropy = entropy(corr_probabilities)
    shannon_entropies.append(shannon_entropy)

    last_date = log_returns.index[end - 1]
    dates.append(last_date)

# Create DataFrames for metrics
metrics_df = pd.DataFrame({
    "Mean Correlation": mean_correlations,
    "Network Entropy": network_entropies,
    "Multilayer Network Entropy": multilayer_network_entropies,
    "Shannon Entropy": shannon_entropies
}, index=dates)

# Plot the time series of metrics
plt.figure(figsize=(14, 12))

# Plot Mean Correlation
plt.subplot(4, 1, 1)
plt.plot(metrics_df.index, metrics_df["Mean Correlation"], marker='o', linestyle='-')
plt.title("Mean Correlation Over Time")
plt.xlabel("Date")
plt.ylabel("Mean Correlation")
plt.grid(True)

# Plot Network Entropy
plt.subplot(4, 1, 2)
plt.plot(metrics_df.index, metrics_df["Network Entropy"], marker='o', linestyle='-', color='#1f77b4')
plt.title("Network Entropy Over Time")
plt.xlabel("Date")
plt.ylabel("Network Entropy")
plt.grid(True)

# Plot Multilayer Network Entropy
plt.subplot(4, 1, 3)
plt.plot(metrics_df.index, metrics_df["Multilayer Network Entropy"], marker='o', linestyle='-', color='g')
plt.title("Multilayer Network Entropy Over Time")
plt.xlabel("Date")
plt.ylabel("Multilayer Network Entropy")
plt.grid(True)

# Plot Shannon Entropy
plt.subplot(4, 1, 4)
plt.plot(metrics_df.index, metrics_df["Shannon Entropy"], marker='o', linestyle='-', color='#ff7f0e')
plt.title("Shannon Entropy Over Time")
plt.xlabel("Date")
plt.ylabel("Shannon Entropy")
plt.grid(True)

plt.tight_layout()
plt.show()
