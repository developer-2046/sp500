import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the distance matrix data (assuming the CSV is the distance matrix)
distance_matrix = pd.read_csv('distance_matrix_final2.csv')

# Assuming the first column is not numeric (if necessary), skip it
distance_matrix = distance_matrix.iloc[:, 1:]  # Skip the first column if necessary

# Convert to numeric, handling errors by coercing to NaN
distance_matrix = distance_matrix.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values (if any)
distance_matrix.dropna(inplace=True)

# Compute eigenvalues and eigenvectors using NumPy
eigenvalues, eigenvectors = np.linalg.eig(distance_matrix)

# Convert eigenvectors to DataFrame for easier manipulation
eigenvectors_df = pd.DataFrame(eigenvectors)

# Set a threshold for flatness (mean absolute value of components below this threshold)
flatness_threshold = 0.01

# List to store indices of flat eigenvectors
flat_eigenvectors = []

# Plot the ranked components of all eigenvectors
plt.figure(figsize=(12, 6))

# Loop through all eigenvectors and plot each one
for i in range(eigenvectors_df.shape[1]):
    # Rank the components of each eigenvector by their absolute values
    ranked_eigenvector = eigenvectors_df.iloc[:, i].abs().sort_values(ascending=False).reset_index(drop=True)

    # Plot the ranked eigenvector
    plt.plot(ranked_eigenvector, label=f'Eigenvector {i+1}', marker='o')

    # Check if the eigenvector is too flat (mean absolute value below threshold)
    if ranked_eigenvector.mean() < flatness_threshold:
        flat_eigenvectors.append(i + 1)  # Store the index (adding 1 for 1-based index)

# Customize the plot
plt.title('Ranked Eigenvector Components (All Eigenvectors)')
plt.xlabel('Component Rank')
plt.ylabel('Absolute Value')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid(True)

# Save the plot as an image
output_dir = "ranked_eigenvector_plots44"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
plt.savefig(os.path.join(output_dir, "ranked_eigenvectors_all.png"))
plt.show()

# Print out the flat eigenvectors
if flat_eigenvectors:
    print(f"Flat eigenvectors (mean absolute value < {flatness_threshold}): {flat_eigenvectors}")
else:
    print("No flat eigenvectors found.")

print(f"Ranked eigenvector plot saved in '{output_dir}/ranked_eigenvectors_all.png'.")
