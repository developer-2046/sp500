import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

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

# Apply noise reduction (power mapping method)
epsilon = 0.6
correlation_matrix_noise_reduced = np.sign(correlation_matrix) * (np.abs(correlation_matrix) ** epsilon)

# Convert the correlation matrix into a distance (similarity) matrix
distance_matrix = np.sqrt(2 * (1 - correlation_matrix_noise_reduced))

# Convert the distance matrix into a squareform matrix
similarity_matrix = squareform(pdist(distance_matrix, 'euclidean'))

# Plot the similarity matrix
plt.figure(figsize=(14, 12))
plt.imshow(similarity_matrix, cmap="coolwarm", aspect='auto')
plt.colorbar()

# Set tick labels
sector_labels = [sectors[symbol] for symbol in correlation_matrix.columns.get_level_values('Symbol')]
unique_sectors = sorted(set(sectors.values()))
sector_to_index = {sector: i for i, sector in enumerate(unique_sectors)}

plt.xticks(range(len(sector_labels)), [sector_to_index[label] for label in sector_labels], rotation=90, fontsize=8)
plt.yticks(range(len(sector_labels)), [sector_to_index[label] for label in sector_labels], fontsize=8)

plt.title("Similarity Matrix of S&P 500 Stocks by GICS Sector", fontsize=16)
plt.show()

print(correlation_matrix.describe())
