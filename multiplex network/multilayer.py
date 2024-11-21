import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load your data
df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date")

# Parse the sector information
sectors = {col.split('_')[0]: col.split('_')[1] for col in df.columns}
df.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], sectors[col.split('_')[0]]) for col in df.columns], names=['Symbol', 'Sector'])

# Calculate log returns
log_returns = np.log(df / df.shift(1))

# Define a threshold for strong correlations
threshold = 0.7

# Initialize the multilayer graph
G = nx.Graph()

# Add intra-layer edges (within sectors)
for sector in df.columns.levels[1]:
    sector_data = log_returns.xs(sector, level='Sector', axis=1)
    correlation_matrix = sector_data.corr()
    for i, stock1 in enumerate(sector_data.columns):
        for j, stock2 in enumerate(sector_data.columns):
            if i != j and correlation_matrix.iloc[i, j] > threshold:
                G.add_edge(stock1[0], stock2[0], weight=correlation_matrix.iloc[i, j])

# Add inter-layer edges (between sectors)
all_correlation_matrix = log_returns.corr()

# Iterate through all possible pairs of stocks
for (stock1, sector1) in all_correlation_matrix.columns:
    for (stock2, sector2) in all_correlation_matrix.columns:
        if sector1 != sector2 and all_correlation_matrix.loc[(stock1, sector1), (stock2, sector2)] > threshold:
            G.add_edge(stock1, stock2, weight=all_correlation_matrix.loc[(stock1, sector1), (stock2, sector2)])

# Define a color map for sectors
sector_colors = {
    'Communication': 'red',
    'Consumer': 'orange',
    'Energy': 'yellow',
    'Financials': 'green',
    'Health': 'blue',
    'Industrials': 'purple',
    'Information': 'cyan',
    'Materials': 'magenta',
    'Real': 'brown',
    'Utilities': 'grey'
}

# Assign colors to nodes based on their sector
node_colors = []
for node in G.nodes:
    symbol = node
    if symbol in sectors:
        sector = sectors[symbol]
        if sector in sector_colors:
            node_colors.append(sector_colors[sector])
        else:
            node_colors.append('black')  # Default color if sector not in predefined colors
    else:
        node_colors.append('black')  # Default color if symbol not found

# Visualization
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(15, 10))
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title('Multilayer Network of S&P 500 Stocks by GICS Sector')
plt.show()
