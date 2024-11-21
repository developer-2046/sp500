# import os
# import pandas as pd
# import numpy as np
# import random
# import matplotlib.pyplot as plt
#
# # Load the eigenvectors data
# eigenvectors = pd.read_csv('eigenvectors.csv')
#
# # Assuming the first column is not numeric, skip it
# eigenvectors = eigenvectors.iloc[:, 1:]  # Skip the first column if necessary
#
# # Convert to numeric, handling errors by coercing to NaN
# eigenvectors = eigenvectors.apply(pd.to_numeric, errors='coerce')
#
# # Drop rows with NaN values (if any)
# eigenvectors.dropna(inplace=True)
#
# # Create a figure for plotting
# plt.figure(figsize=(12, 6))
#
# # Loop through each eigenvector column
# for i in range(eigenvectors.shape[1]):
#     # Extract the eigenvector
#     eigenvector = eigenvectors.iloc[:, i]
#
#     # Rank the components of the eigenvector by their absolute values
#     ranked_eigenvector = eigenvector.abs().sort_values(ascending=False).reset_index(drop=True)
#
#     # Plot the ranked eigenvector components
#     plt.plot(ranked_eigenvector, label=f'Eigenvector {i + 1}', marker='o')
#
# # Add plot details
# plt.title('Ranked Eigenvector Components')
# plt.xlabel('Component Rank')
# plt.ylabel('Absolute Value')
# plt.legend()
# plt.grid(True)
#
# # Save the plot as an image
# output_dir = "ranked_eigenvector_plots"
# os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
# plt.savefig(os.path.join(output_dir, "all_ranked_eigenvectors.png"))
# plt.show()
#
# print(f"Ranked eigenvector plot saved in '{output_dir}/all_ranked_eigenvectors.png'.")
#---------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the distance matrix
distance_matrix = pd.read_csv('distance_matrix.csv')

# Check if the matrix is square
print(f"Shape of the matrix: {distance_matrix.shape}")

if distance_matrix.shape[0] != distance_matrix.shape[1]:
    print("Matrix is not square. Using covariance matrix for eigenvalue decomposition.")
    # Calculate the covariance matrix (which will be square)
    cov_matrix = np.cov(distance_matrix, rowvar=False)
else:
    # If it's already square, we can directly use the matrix
    print("Matrix is square, using it directly.")
    cov_matrix = distance_matrix.values

# Function to calculate the eigenvector corresponding to lambda max
def calculate_eigen(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    lambda_max_index = np.argmax(np.real(eigenvalues))  # Index of the max eigenvalue
    return eigenvectors[:, lambda_max_index]  # Corresponding eigenvector

# Calculate the eigenvector corresponding to the largest eigenvalue
eigenvector = calculate_eigen(cov_matrix)

# Directory to save the plot
output_dir = "eigen_vector_plots3"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Plot the eigenvector and save as image
plt.figure(figsize=(10, 6))
plt.plot(eigenvector, marker='o')
plt.title(f'Eigenvector corresponding to lambda max')
plt.xlabel('Index')
plt.ylabel('Eigenvector Value')
plt.grid(True)

# Save the plot
plot_filename = f"eigenvector_plot.png"
plt.savefig(os.path.join(output_dir, plot_filename))
plt.close()  # Close the plot to free memory

print(f"Eigenvector plot has been saved in the '{output_dir}' directory.")

