import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Load the data
df = pd.read_csv('SP500_25May24_cleaned_filled.csv', index_col="Date")

# Parse the sector information from the column names
sectors = {col.split('_')[0]: col.split('_')[1] for col in df.columns}
df.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], sectors[col.split('_')[0]]) for col in df.columns], names=['Symbol', 'Sector'])

# Sort columns by sector
df = df.sort_index(axis=1, level=1)

# Calculate log returns
log_returns = np.log(df / df.shift(1)).dropna()

# Calculate the correlation matrix
correlation_matrix = log_returns.corr()

# Threshold the correlation matrix to create the adjacency matrix
threshold = 0.7
adjacency_matrix = np.where(np.abs(correlation_matrix) > threshold, 1, 0)

# Create a NetworkX graph from the adjacency matrix
G = nx.from_numpy_array(adjacency_matrix)

# Plot the adjacency matrix with blue and green colors
plt.figure(figsize=(14, 12))
plt.imshow(adjacency_matrix, cmap=plt.get_cmap('winter', 2), aspect='auto')
plt.colorbar(ticks=[0, 1], label='Adjacency (1: Connected, 0: Not Connected)')
plt.clim(-0.5, 1.5)

# Set tick labels
sector_labels = [sectors[symbol] for symbol in correlation_matrix.columns.get_level_values('Symbol')]
unique_sectors = sorted(set(sectors.values()))
sector_to_index = {sector: i for i, sector in enumerate(unique_sectors)}

plt.xticks(range(len(sector_labels)), [sector_to_index[label] for label in sector_labels], rotation=90, fontsize=8)
plt.yticks(range(len(sector_labels)), [sector_to_index[label] for label in sector_labels], fontsize=8)

plt.title("Adjacency Matrix of S&P 500 Stocks by GICS Sector", fontsize=16)
plt.show()

# Define a color map for sectors
sector_to_color = {sector: plt.cm.tab20(i / len(unique_sectors)) for i, sector in enumerate(unique_sectors)}

# Create a list of node colors based on sector
node_colors = [sector_to_color[sectors[symbol]] for symbol in correlation_matrix.columns.get_level_values('Symbol')]

# Plot the graph
plt.figure(figsize=(14, 12))
pos = nx.spring_layout(G, seed=42)  # positions for all nodes, fixed seed for reproducibility

# Draw nodes with colors
nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors, edgecolors='k', linewidths=0.5)

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.3)

# Draw labels
labels = {i: symbol for i, symbol in enumerate(correlation_matrix.columns.get_level_values('Symbol'))}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

# Add legend for sectors
for sector, color in sector_to_color.items():
    plt.scatter([], [], c=[color], label=sector, edgecolors='k', linewidths=0.5)
plt.legend(scatterpoints=1, frameon=True, labelspacing=1, loc='best', title='Sectors')

plt.title("Graph", fontsize=16)
plt.axis('off')
plt.show()
