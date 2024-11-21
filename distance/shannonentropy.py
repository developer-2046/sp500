# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from fontTools.ttLib.woff2 import base128Size
# from scipy.stats import entropy
#
# # Load the distance matrix
# distance_matrix = pd.read_csv('distance_matrix_final2.csv', index_col=0)
#
# # Drop any unnecessary columns and ensure numeric data
# if 'Unnamed: 0' in distance_matrix.columns:
#     distance_matrix.drop(columns=['Unnamed: 0'], inplace=True)
# distance_matrix = distance_matrix.apply(pd.to_numeric, errors='coerce')
# distance_matrix.dropna(axis=1, inplace=True)
# distance_matrix.dropna(axis=0, inplace=True)
#
# # Calculate eigenvalues and eigenvectors
# eigenvalues, eigenvectors = np.linalg.eig(distance_matrix.values)
#
# # Convert eigenvectors array to a DataFrame
# df = pd.DataFrame(eigenvectors)
#
# # Function to calculate entropy
# def calculate_entropy(vector):
#     if len(vector) == 0 or np.all(np.isnan(vector)):
#         return np.nan
#
#     prob_vector = np.abs(vector) / np.sum(np.abs(vector))
#
#     return entropy(prob_vector)
#
# # Calculate entropy for each eigenvector (applies function column-wise)
# entropies = df.apply(calculate_entropy, axis=0)
#
# # Plot the entropy values
# # Plot the Shannon entropy values
# plt.figure(figsize=(10, 6))
# plt.plot(entropies.index, entropies.values, marker='o')
# plt.title('Shannon Entropy of Eigenvectors')
# plt.xlabel('Eigenvector Index')
# plt.ylabel('Shannon Entropy')
# plt.grid(True)
# plt.show()
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Load the distance matrix
distance_matrix = pd.read_csv('../RMTpaper/distance_matrix_final2.csv', index_col=0)

# Drop any unnecessary columns and ensure numeric data
if 'Unnamed: 0' in distance_matrix.columns:
    distance_matrix.drop(columns=['Unnamed: 0'], inplace=True)
distance_matrix = distance_matrix.apply(pd.to_numeric, errors='coerce')
distance_matrix.dropna(axis=1, inplace=True)
distance_matrix.dropna(axis=0, inplace=True)

# Flatten the entire distance matrix into a 1D array
flattened_matrix = distance_matrix.values.flatten()

# Remove any NaN or zero values to avoid issues in entropy calculation
flattened_matrix = flattened_matrix[~np.isnan(flattened_matrix)]
flattened_matrix = flattened_matrix[flattened_matrix > 0]  # Remove zeros if applicable

# Normalize the flattened matrix to create a probability distribution
prob_vector = flattened_matrix / np.sum(flattened_matrix)

# Calculate Shannon entropy for the entire distance matrix
matrix_entropy = entropy(prob_vector)

# Print the entropy value
print(f"Shannon Entropy for the entire distance matrix: {matrix_entropy}")

# Plot the single Shannon entropy value (for demonstration, we plot it as a single point)
# plt.figure(figsize=(5, 5))
# plt.scatter(1, matrix_entropy, color='blue', marker='o')
# plt.title('Shannon Entropy of the Distance Matrix')
# plt.xlabel('Distance Matrix')
# plt.ylabel('Shannon Entropy')
# plt.xticks([1], ['Distance Matrix'])
# plt.grid(True)
# plt.show()

