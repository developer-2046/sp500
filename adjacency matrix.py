import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the data with date parsing
df = pd.read_csv('SP500_25May24_cleaned_filled.csv', index_col="Date", parse_dates=True)

# Parse the sector information from the column names
sectors = {col.split('_')[0]: col.split('_')[1] for col in df.columns}
df.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], sectors[col.split('_')[0]]) for col in df.columns], names=['Symbol', 'Sector'])

# Sort columns by sector
df = df.sort_index(axis=1, level=1)

# Calculate log returns
log_returns = np.log(df / df.shift(1))

# Calculate the mean correlation matrix
mean_correlation_matrix = log_returns.corr()

# Define the threshold
threshold = 0.6

# Create the adjacency matrix based on the threshold
adjacency_matrix = np.zeros(mean_correlation_matrix.shape)

# Populate the adjacency matrix
for i in range(mean_correlation_matrix.shape[0]):
    for j in range(mean_correlation_matrix.shape[1]):
        if mean_correlation_matrix.iloc[i, j] > threshold:
            adjacency_matrix[i, j] = 1

# Convert adjacency matrix to a DataFrame for better readability
adjacency_df = pd.DataFrame(adjacency_matrix, index=mean_correlation_matrix.index, columns=mean_correlation_matrix.columns)

# Save the adjacency matrix to a CSV file
adjacency_df.to_csv('adjacency_matrix.csv')

# Display the adjacency matrix
print(adjacency_df)

# Optionally, you can visualize the graph using NetworkX
G = nx.from_numpy_array(adjacency_matrix)
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
plt.show()

