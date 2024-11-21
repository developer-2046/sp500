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

# Plot the ranked components of only eigenvector 24 and 25
plt.figure(figsize=(12, 6))

# Selectively plot eigenvector 24 (index 23) and eigenvector 25 (index 24)
for i in [35, 36]:  # 23 for eigenvector 24, 24 for eigenvector 25
    # Rank the components of each eigenvector by their absolute values
    ranked_eigenvector = eigenvectors_df.iloc[:, i].abs().sort_values(ascending=False).reset_index(drop=True)

    # Plot the ranked eigenvector
    plt.plot(ranked_eigenvector, label=f'Eigenvector {i+1}', marker='o')

# Customize the plot
plt.title('Ranked Eigenvector Components (Eigenvectors 24 and 25)')
plt.xlabel('Component Rank')
plt.ylabel('Absolute Value')
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.grid(True)

# Save the plot as an image
output_dir = "ranked_eigenvector_plots"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
plt.savefig(os.path.join(output_dir, "ranked_eigenvectors_24_25.png"))
plt.show()

print(f"Ranked eigenvector plot saved in '{output_dir}/ranked_eigenvectors_24_25.png'.")
