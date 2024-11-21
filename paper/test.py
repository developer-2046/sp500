import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Load SP500 data and calculate returns
df = pd.read_csv('SP500_25May24_cleaned_filled.csv', index_col=0)
df = df.pct_change().dropna()

# Set rolling window size
window_size = 20

# Prepare file for saving
output_file = 'wishart_matrices.csv'
with open(output_file, 'w') as f:
    f.write("MatrixID,Row,Col,Value\n")  # Initialize CSV header for easy reading

matrix_id = 0  # ID for each Wishart matrix

# Compute and save Wishart matrices
for start in range(len(df) - window_size + 1):
    # Get data for the current window
    window_data = df.iloc[start:start + window_size]

    # Calculate the pairwise distance matrix
    distance_matrix = squareform(pdist(window_data.T, metric='euclidean'))

    # Center the distance matrix
    centered_distance_matrix = distance_matrix - np.mean(distance_matrix, axis=0)

    # Calculate the Wishart matrix
    wishart_matrix = centered_distance_matrix.T @ centered_distance_matrix

    # Save the Wishart matrix to the CSV file
    with open(output_file, 'a') as f:
        for i in range(wishart_matrix.shape[0]):
            for j in range(wishart_matrix.shape[1]):
                f.write(f"{matrix_id},{i},{j},{wishart_matrix[i, j]}\n")
    matrix_id += 1  # Increment the ID for the next matrix
