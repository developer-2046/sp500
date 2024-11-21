import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date")

# Parse the sector information from the column names
sectors = {col.split('_')[0]: col.split('_')[1] for col in df.columns}
df.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], sectors[col.split('_')[0]]) for col in df.columns], names=['Symbol', 'Sector'])

# Sort columns by sector
df = df.sort_index(axis=1, level=1)

# Fill missing values using forward fill
df.ffill(inplace=True)

# Calculate log returns
log_returns = np.log(df / df.shift(1))

# Drop NaN rows resulting from shift
log_returns.dropna(inplace=True)

def is_valid_eigenvector(eigenvector):
    # Check if all values are positive or all are negative
    all_positive = np.all(eigenvector >= 0)
    all_negative = np.all(eigenvector <= 0)
    return all_positive or all_negative

# Define a function to plot the eigenvector corresponding to λ_max
def plot_eigenvector(eigenvector, index, title, output_dir):
    #validate
    if not is_valid_eigenvector(eigenvector):
        print(f"Skipping plot for {index}: Invalid eigenvector with mixed positive and negative values.")
        return
    plt.figure(figsize=(10, 6))
    plt.plot(eigenvector, marker='o')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Eigenvector Value')
    plt.grid(True)
    # Save plot
    plot_filename = os.path.join(output_dir, f"eigenvector_lambda_max_{index}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot for {index}: {plot_filename}")

# Define a function to compute the distance matrix from the correlation matrix
def compute_distance_matrix(correlation_matrix):
    return np.sqrt(2 * (1 - correlation_matrix))

# Directory to save eigenvector plots
output_dir = "eigen_vector_plots_lambda_max_windows"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Number of days per window
window_size = 20

# Iterate over the data in windows of `window_size`
for start in range(0, len(log_returns) - window_size + 1, window_size):
    end = start + window_size
    window_data = log_returns.iloc[start:end]

    # Calculate the correlation matrix for this window
    correlation_matrix = window_data.corr()

    # Calculate the distance matrix for this window
    distance_matrix = compute_distance_matrix(correlation_matrix)

    # Drop any rows or columns with NaN values in the distance matrix
    distance_matrix.dropna(axis=1, inplace=True)
    distance_matrix.dropna(axis=0, inplace=True)

    # Ensure the distance matrix is square before calculating eigenvalues
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        print(f"Skipping non-square matrix for window ending {log_returns.index[end - 1]}.")
        continue

    # Calculate eigenvalues and eigenvectors for the distance matrix
    eigenvalues, eigenvectors = np.linalg.eig(distance_matrix.values)

    # Get the index of the largest eigenvalue (λ_max)
    lambda_max_index = np.argmax(np.real(eigenvalues))

    # Get the eigenvector corresponding to λ_max
    eigenvector_lambda_max = eigenvectors[:, lambda_max_index]

    # Get the last date of the current window for title purposes
    last_date = log_returns.index[end - 1]

    # Plot and save the eigenvector corresponding to λ_max for the current window
    plot_title = f'Eigenvector corresponding to λ_max - Window ending {last_date}'
    plot_eigenvector(eigenvector_lambda_max, last_date, plot_title, output_dir)

print(f"All valid eigenvector plots corresponding to λ_max have been saved in the '{output_dir}' directory.")
