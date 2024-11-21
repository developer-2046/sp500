import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the eigenvectors data, ensuring all data is numeric
eigenvectors = pd.read_csv('eigenvectors.csv')

# Exclude the first column if it is non-numeric (e.g., names, indices)
eigenvectors = eigenvectors.iloc[:, 1:]  # Skip the first column

# Convert remaining columns to numeric, forcing errors to NaN
eigenvectors = eigenvectors.apply(pd.to_numeric, errors='coerce')

# Print the first few rows to ensure data is loaded correctly
print("Eigenvectors Data (First 5 rows):")
print(eigenvectors.head())

# Function to calculate Shannon entropy
def calculate_entropy(vector):
    prob_vector = np.abs(vector) / np.sum(np.abs(vector))  # Normalize to get probabilities
    entropy = -np.sum(prob_vector * np.log2(prob_vector + 1e-12))  # Add small epsilon to avoid log(0)
    return entropy

# Drop rows with NaN values (if any)
eigenvectors.dropna(inplace=True)

# Calculate Shannon entropy for each row (time point)
entropies = eigenvectors.apply(calculate_entropy, axis=1)

# Print the calculated entropies
print("Calculated Entropies:")
print(entropies)

# Create a time series plot for Shannon entropy
plt.figure(figsize=(12, 6))
plt.plot(entropies, marker='o')
plt.title('Shannon Entropy Time Series')
plt.xlabel('Time Point')
plt.ylabel('Shannon Entropy')
plt.grid(True)

# Save the plot as an image
output_dir = "sha_entr_plt"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
plt.savefig(os.path.join(output_dir, "shannon_entropy_ts.png"))
plt.show()

print(f"Shannon entropy time series plot saved in '{output_dir}/shannon_entropy_ts.png'.")



