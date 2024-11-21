import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import entropy

# Load the data
df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date")

# Parse the sector information from the column names
sectors = {col.split('_')[0]: col.split('_')[1] for col in df.columns}
df.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], sectors[col.split('_')[0]]) for col in df.columns], names=['Symbol', 'Sector'])

# Sort columns by sector
df = df.sort_index(axis=1, level=1)

# Calculate log returns
log_returns = np.log(df / df.shift(1))

# Get unique sector codes for labeling
unique_sectors = sorted(set(sectors.values()))

# Create a mapping from sector code to index for labeling
sector_to_index = {sector: i for i, sector in enumerate(unique_sectors)}

# Number of days per window
window_size = 20

# Initialize lists to store entropy values and dates
network_entropies = []
shannon_entropies = []
window_dates = []

# Iterate over the data in windows of `window_size`
for start in range(0, len(log_returns) - window_size + 1, window_size):
    end = start + window_size
    window_data = log_returns.iloc[start:end]
    correlation_matrix = window_data.corr()
    last_date = log_returns.index[end - 1]  # Get the last date of the current window

    # Calculate network entropy
    G = nx.Graph()
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            if np.abs(correlation_matrix.iloc[i, j]) > 0.7:  # Threshold of 0.7
                G.add_edge(i, j, weight=correlation_matrix.iloc[i, j])

    degrees = np.array([deg for _, deg in G.degree()])
    unique, counts = np.unique(degrees, return_counts=True)
    degree_probabilities = counts / counts.sum()
    network_entropy = entropy(degree_probabilities)
    network_entropies.append(network_entropy)

    # Calculate Shannon entropy
    corr_values = correlation_matrix.values.flatten()
    corr_values = corr_values[corr_values != 1]  # Exclude self-correlation
    corr_probabilities = np.abs(corr_values) / np.sum(np.abs(corr_values))
    shannon_entropy = entropy(corr_probabilities)
    shannon_entropies.append(shannon_entropy)
# find reference
    window_dates.append(last_date)

# Plot network entropy and Shannon entropy
plt.figure(figsize=(14, 8))

# Plot Network Entropy
plt.subplot(2, 1, 1)
plt.plot(window_dates, network_entropies, label='Network Entropy', color='#1f77b4', marker='o')
plt.title('Network Entropy Over Time')
plt.xlabel('Date')
plt.ylabel('Network Entropy')
plt.grid(True)

# Plot Shannon Entropy
plt.subplot(2, 1, 2)
plt.plot(window_dates, shannon_entropies, label='Shannon Entropy', color='#ff7f0e', marker='o')
plt.title('Shannon Entropy Over Time')
plt.xlabel('Date')
plt.ylabel('Shannon Entropy')
plt.grid(True)

plt.tight_layout()
plt.show()
