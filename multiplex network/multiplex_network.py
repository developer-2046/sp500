import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Load the data
df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date")

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
G = nx.Graph()

# Add nodes with sector information
for i, symbol in enumerate(correlation_matrix.columns.get_level_values('Symbol')):
    G.add_node(i, symbol=symbol, sector=sectors[symbol])

# Add intra-layer (same sector) edges
for i in range(adjacency_matrix.shape[0]):
    for j in range(i + 1, adjacency_matrix.shape[1]):
        if adjacency_matrix[i, j] == 1 and G.nodes[i]['sector'] == G.nodes[j]['sector']:
            G.add_edge(i, j, weight=correlation_matrix.iloc[i, j], color='blue')

# Add inter-layer (different sectors) edges
for i in range(adjacency_matrix.shape[0]):
    for j in range(i + 1, adjacency_matrix.shape[1]):
        if adjacency_matrix[i, j] == 1 and G.nodes[i]['sector'] != G.nodes[j]['sector']:
            G.add_edge(i, j, weight=correlation_matrix.iloc[i, j], color='green')

# Draw the multiplex network
pos = nx.kamada_kawai_layout(G)  # Kamada-Kawai layout for all nodes

# Draw nodes
sector_colors = {
    'Communication': 'purple', 'Consumer': 'orange', 'Energy': 'red', 'Financials': 'yellow',
    'Health': 'pink', 'Industrials': 'brown', 'Information': 'cyan', 'Materials': 'green',
    'Real': 'blue', 'Utilities': 'grey'
}
node_colors = [sector_colors[G.nodes[node]['sector']] for node in G.nodes]

plt.figure(figsize=(14, 12))

# Draw intra-layer edges
intra_edges = [edge for edge in G.edges(data=True) if edge[2]['color'] == 'blue']
nx.draw_networkx_edges(G, pos, edgelist=intra_edges, edge_color='blue', alpha=0.5)

# Draw inter-layer edges
inter_edges = [edge for edge in G.edges(data=True) if edge[2]['color'] == 'green']
nx.draw_networkx_edges(G, pos, edgelist=inter_edges, edge_color='green', alpha=0.5)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors)

# Draw labels
labels = {i: G.nodes[i]['symbol'] for i in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=8)

# Add a legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=sector_colors[sector], markersize=10) for sector in sector_colors]
plt.legend(handles, sector_colors.keys(), title="Sectors", loc="best")

plt.title("Multiplex Network of S&P 500 Stocks by GICS Sector", fontsize=16)
plt.show()
