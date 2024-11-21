import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Load SP500 data and calculate returns
df = pd.read_csv('SP500_25May24_cleaned_filled.csv', index_col=0)
df = df.pct_change().dropna()

# Set rolling window size
window_size = 500

# Lists to store results
distance_matrices = []
wishart_matrices = []
eigenvalues_list = []
eigenvectors_list = []

for start in range(len(df) - window_size + 1):
    # Get data for the current window
    window_data = df.iloc[start:start + window_size]

    # Calculate the pairwise distance matrix between stocks over this window
    distance_matrix = squareform(pdist(window_data.T, metric='euclidean'))
    distance_matrices.append(distance_matrix)

    # Center the distance matrix and compute Wishart-like matrix
    centered_distance_matrix = distance_matrix - np.mean(distance_matrix, axis=0)
    wishart_matrix = centered_distance_matrix.T @ centered_distance_matrix
    wishart_matrices.append(wishart_matrix)

    # Calculate eigenvalues and eigenvectors of the Wishart matrix
    eigenvalues, eigenvectors = np.linalg.eig(wishart_matrix)

    # Store the results
    eigenvalues_list.append(eigenvalues)
    eigenvectors_list.append(eigenvectors)

# Save eigenvalues and eigenvectors to CSV files for further analysis
# Saving eigenvalues


# Saving eigenvectors as a flattened format
eigenvectors_df = pd.DataFrame([vec.flatten() for vec in eigenvectors_list])
eigenvectors_df.to_csv('eigenvectors.csv', index=False)

# Plot the last Wishart matrix as an example
wishart_matrix_to_plot = wishart_matrices[-1]
plt.figure(figsize=(10, 8))
sns.heatmap(wishart_matrix_to_plot, cmap="viridis", annot=False, cbar=True)
plt.title("Wishart Matrix Derived from SP500 Distance Matrix (20-day window)")
plt.xlabel("Stocks")
plt.ylabel("Stocks")
plt.show()
