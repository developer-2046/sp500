import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the distance matrix
distance_matrix = pd.read_csv('../RMTpaper/distance_matrix_final2.csv', index_col=0)

# Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in distance_matrix.columns:
    distance_matrix.drop(columns=['Unnamed: 0'], inplace=True)

# Convert all values to numeric, forcing non-numeric values to NaN
distance_matrix = distance_matrix.apply(pd.to_numeric, errors='coerce')

# Drop rows or columns with NaN values
distance_matrix.dropna(axis=1, inplace=True)  # Drop columns with NaN
distance_matrix.dropna(axis=0, inplace=True)  # Drop rows with NaN

# Calculate eigenvalues and eigenvectors for the entire matrix
eigenvalues, eigenvectors = np.linalg.eig(distance_matrix.values)

# Get the index of the largest eigenvalue (lambda max)
lambda_max_index = np.argmax(np.real(eigenvalues))

# Get the eigenvector corresponding to the largest eigenvalue (λ_max)
eigenvector_lambda_max = eigenvectors[:, lambda_max_index]

# Directory to save the plots
output_dir = "eigen_vector_plots_lambda_max23"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Plot each row of the eigenvector corresponding to λ_max and save as image
for i, stock in enumerate(distance_matrix.index):
    plt.figure(figsize=(10, 6))
    plt.plot(eigenvector_lambda_max, marker='o')
    plt.title(f'Eigenvector corresponding to λ_max - {stock}')
    plt.xlabel('Index')
    plt.ylabel('Eigenvector Value')
    plt.grid(True)

    # Save the plot
    plot_filename = f"eigenvector_lambda_max_{stock}.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()  # Close the plot to free memory

    print(f"Saved plot for {stock}: {plot_filename}")

print(f"All eigenvector plots corresponding to λ_max have been saved in the '{output_dir}' directory.")


# #------------------------------------------------------------------------------------ try 2
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# # Load the distance matrix
# distance_matrix = pd.read_csv('distance_matrix.csv')
#
# # Function to calculate the eigenvector corresponding to lambda max
# def calculate_eigen(vector):
#     matrix = np.outer(vector, vector)  # Create a square matrix from the vector
#     eigenvalues, eigenvectors = np.linalg.eig(matrix)
#     lambda_max_index = np.argmax(np.real(eigenvalues))  # Index of the max eigenvalue
#     return eigenvectors[:, lambda_max_index]  # Corresponding eigenvector
#
# # Calculate eigenvectors for each row
# eigenvectors = [calculate_eigen(row) for row in distance_matrix.values]
#
# # Directory to save the plots
# output_dir = "eigen_vector_plots2"
# os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
#
# # Plot each eigenvector and save as image
# for i, eigenvector in enumerate(eigenvectors):
#     # Check if the eigenvector is all zeros
#     if np.all(eigenvector == 0):
#         print(f"Eigenvector for Plot {i + 1} is all zeros, skipping plot.")
#         continue
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(eigenvector, marker='o')
#     plt.title(f'Eigenvector corresponding to lambda max - Plot {i + 1}')
#     plt.xlabel('Index')
#     plt.ylabel('Eigenvector Value')
#     plt.grid(True)
#
#     # Save the plot
#     plot_filename = f"eigenvector_plot_{i + 1}.png"
#     plt.savefig(os.path.join(output_dir, plot_filename))
#     plt.close()  # Close the plot to free memory
#
# print(f"All eigenvector plots have been saved in the '{output_dir}' directory.")
#-----------------------------------------

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# # Load the distance matrix
# distance_matrix = pd.read_csv('distance_matrix_final.csv', header=None)
#
# # Convert all values in the matrix to numeric to avoid string issues
# distance_matrix = distance_matrix.apply(pd.to_numeric, errors='coerce')
#
# # Check for any NaN values (which may have been created during the conversion process)
# if distance_matrix.isnull().values.any():
#     print("Warning: There are NaN values in the distance matrix after conversion.")
#     print(distance_matrix.isnull().sum())
#     # Optionally, fill NaN values with 0 or handle them appropriately
#     distance_matrix.fillna(0, inplace=True)
#
# # Function to calculate the eigenvector corresponding to lambda max
# def calculate_eigen(vector):
#     matrix = np.outer(vector, vector)  # Create a square matrix from the vector
#     eigenvalues, eigenvectors = np.linalg.eig(matrix)
#     lambda_max_index = np.argmax(np.real(eigenvalues))  # Index of the max eigenvalue
#     return eigenvectors[:, lambda_max_index]  # Corresponding eigenvector
#
# # Calculate eigenvectors for each row
# eigenvectors = [calculate_eigen(row) for row in distance_matrix.values]
#
# # Directory to save the plots
# output_dir = "eigen_vector_plots5"
# os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
#
# # Plot each eigenvector and save as image
# for i, eigenvector in enumerate(eigenvectors):
#     # Check if the eigenvector is all zeros
#     if np.all(eigenvector == 0):
#         print(f"Eigenvector for Plot {i + 1} is all zeros, skipping plot.")
#         continue
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(eigenvector, marker='o')
#     plt.title(f'Eigenvector corresponding to lambda max - Plot {i + 1}')
#     plt.xlabel('Index')
#     plt.ylabel('Eigenvector Value')
#     plt.grid(True)
#
#     # Save the plot
#     plot_filename = f"eigenvector_plot_{i + 1}.png"
#     plt.savefig(os.path.join(output_dir, plot_filename))
#     plt.close()  # Close the plot to free memory
#
# print(f"All eigenvector plots have been saved in the '{output_dir}' directory.")
#------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# # Load the distance matrix, treating the first column as index
# distance_matrix = pd.read_csv('distance_matrix_final2.csv', index_col=0)
#
# # Print the first few rows to check the data
# print(distance_matrix.head())
#
# # Ensure the matrix contains only numeric values
# distance_matrix = distance_matrix.apply(pd.to_numeric, errors='coerce')
#
# # Check for NaN values (which could indicate non-numeric columns)
# print("Checking for NaN values after conversion:")
# print(distance_matrix.isnull().sum())
#
# # Drop any columns or rows that contain NaN values
# distance_matrix.dropna(axis=1, inplace=True)  # Drop non-numeric columns
# distance_matrix.dropna(axis=0, inplace=True)  # Drop non-numeric rows
#
# # Ensure that the matrix is square after removing NaNs
# if distance_matrix.shape[0] != distance_matrix.shape[1]:
#     raise ValueError(f"Matrix is not square after cleanup: {distance_matrix.shape}")
#
# # Convert the matrix to a NumPy array for computation
# matrix_values = distance_matrix.values
#
# # Function to calculate the eigenvector corresponding to lambda max
# def calculate_eigen(matrix):
#     eigenvalues, eigenvectors = np.linalg.eig(matrix)
#     lambda_max_index = np.argmax(np.real(eigenvalues))  # Index of the max eigenvalue
#     return eigenvectors[:, lambda_max_index]  # Corresponding eigenvector
#
# # Calculate the eigenvector
# eigenvector = calculate_eigen(matrix_values)
#
# # Directory to save the plot
# output_dir = "eigen_vector_plots6"
# os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
#
# # Plot the eigenvector and save as image
# plt.figure(figsize=(10, 6))
# plt.plot(eigenvector, marker='o')
# plt.title(f'Eigenvector corresponding to lambda max')
# plt.xlabel('Index')
# plt.ylabel('Eigenvector Value')
# plt.grid(True)
#
# # Save the plot
# plot_filename = f"eigenvector_plot.png"
# plt.savefig(os.path.join(output_dir, plot_filename))
# plt.close()  # Close the plot to free memory
#
# print(f"Eigenvector plot has been saved in the '{output_dir}' directory.")
#--working--#
#----------------------------------------------------------------------------------




