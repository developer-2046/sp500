import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Step 1: Load your SP500 data and calculate returns
# Ensure your data has rows as dates and columns as stocks

# Example: Load SP500 data (you might need to replace this with your file path)
# The data should be prices, so we calculate returns from prices
df = pd.read_csv('SP500_25May24_cleaned_filled.csv', index_col=0)
df = df.pct_change().dropna()  # Calculate daily returns and drop NaN values

# Step 2: Set the rolling window size for analysis
window_size = 1000  # Use a 20-day rolling window, or adjust as needed

# Step 3: Calculate the rolling distance and Wishart matrices
distance_matrices = []
wishart_matrices = []

for start in range(len(df) - window_size + 1):
    # Get data for the current window
    window_data = df.iloc[start:start + window_size]

    # Calculate the pairwise distance matrix between stocks over this window
    # Using Euclidean distance as the metric for distances between stocks
    distance_matrix = squareform(pdist(window_data.T, metric='euclidean'))
    distance_matrices.append(distance_matrix)

    # Step 4: Generate a Wishart-like matrix from the distance matrix
    # Center the distance matrix by subtracting the mean
    centered_distance_matrix = distance_matrix - np.mean(distance_matrix, axis=0)

    # Compute the Wishart matrix by taking the dot product of the centered matrix with itself
    wishart_matrix = centered_distance_matrix.T @ centered_distance_matrix
    wishart_matrices.append(wishart_matrix)

# Step 5: Plot the Wishart matrix from the last rolling window as an example
wishart_matrix_to_plot = wishart_matrices[-1]  # Choose the most recent Wishart matrix

# Plot the Wishart matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(wishart_matrix_to_plot, cmap="viridis", annot=False, cbar=True)
plt.title("Wishart Matrix Derived from SP500 Distance Matrix (20-day window)")
plt.xlabel("Stocks")
plt.ylabel("Stocks")
plt.show()
