import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

def calc(prob_vector):
    entropy = 0
    for p in prob_vector:
        if p > 0:  # Shannon entropy is undefined for p=0
            entropy -= p * np.log(np.sqrt(p))  # Natural log used here
    return entropy

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

# Define a function to compute the distance matrix from the correlation matrix
def compute_distance_matrix(correlation_matrix):
    return np.sqrt(2 * (1 - correlation_matrix))

# Function to calculate Shannon entropy of a matrix
def calculate_shannon_entropy(matrix):
    # Flatten the matrix into a 1D array
    flattened_matrix = matrix.values.flatten()

    # Remove any NaN or zero values to avoid issues in entropy calculation
    flattened_matrix = flattened_matrix[~np.isnan(flattened_matrix)]
    flattened_matrix = flattened_matrix[flattened_matrix > 0]  # Remove zeros if applicable

    # Normalize the flattened matrix to create a probability distribution
    prob_vector = flattened_matrix / np.sum(flattened_matrix)

    # Calculate Shannon entropy for the matrix
    return calc(prob_vector)

# Directory to save eigenvector plots
output_dir = "eigen_vector_plots_lambda_max_windows"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Number of days per window
window_size = 20

# List to store Shannon entropy values and their corresponding window end dates
entropy_values = []
window_end_dates = []

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

    # Ensure the distance matrix is square before calculating entropy
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        print(f"Skipping non-square matrix for window ending {log_returns.index[end - 1]}.")
        continue

    # Calculate Shannon entropy for the distance matrix
    matrix_entropy = calculate_shannon_entropy(distance_matrix)

    # Append the entropy value and the corresponding end date to the lists
    entropy_values.append(matrix_entropy)
    window_end_dates.append(log_returns.index[end - 1])

# Convert the results into a DataFrame for easier plotting
entropy_df = pd.DataFrame({'Date': window_end_dates, 'Shannon Entropy': entropy_values})
entropy_df.set_index('Date', inplace=True)

# Plot the Shannon entropy time series
plt.figure(figsize=(10, 6))
plt.plot(entropy_df.index, entropy_df['Shannon Entropy'], marker='o', linestyle='-', color='b')
plt.title('Shannon Entropy of Distance Matrix Over Time')
plt.xlabel('Date')
plt.ylabel('Shannon Entropy')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Print success message
print("Shannon entropy time series has been plotted.")
