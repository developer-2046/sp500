# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the data
# df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date")
#
# # Extract stock names from the columns
# stock_names = df.columns.get_level_values(0).unique()
#
# # Load the distance matrix (assuming it's saved as 'distance_matrix.csv')
# distance_matrix = np.loadtxt('distance_matrix.csv', delimiter=',')
#
# # Calculate eigenvalues and eigenvectors
# eigenvalues, eigenvectors = np.linalg.eig(distance_matrix)
#
# # Randomly select 5 stocks (indices)
# np.random.seed(43)  # For reproducibility
# selected_indices = np.random.choice(eigenvectors.shape[0], 5, replace=False)
#
# # Plot the icon vectors (eigenvectors) for the selected stocks
# plt.figure(figsize=(10, 12))
#
# for i, index in enumerate(selected_indices):
#     plt.subplot(5, 1, i + 1)
#     plt.plot(eigenvectors[:, index])
#     plt.title(f'Icon Vector for {stock_names[index]}')
#     plt.grid(True)
#
# plt.tight_layout()
# plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the data
# df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date")
#
# # Extract stock names from the columns
# stock_names = df.columns.get_level_values(0).unique()
#
# # Load the distance matrix (assuming it's saved as 'distance_matrix.csv')
# distance_matrix = np.loadtxt('distance_matrix.csv', delimiter=',')
#
# # Calculate eigenvalues and eigenvectors
# eigenvalues = pd.DataFrame(np.linalg.eig(distance_matrix))
#
# #eigenvalues.to_csv('eigenvalues.csv')
#
# # Calculate the maximum value in each icon vector (eigenvector)
#
# real_parts = np.real(eigenvalues)
# imaginary_parts = np.imag(eigenvalues)
#
# # Plotting
# plt.scatter(real_parts, imaginary_parts, color='blue', marker='o')
# plt.title('Eigenvalues')
# plt.xlabel('Real Part')
# plt.ylabel('Imaginary Part')
# plt.grid(True)
# plt.axhline(0, color='black',linewidth=0.5)
# plt.axvline(0, color='black',linewidth=0.5)
# plt.show()
#
# #Plot the maximum icon values for all stocks
# # plt.figure(figsize=(10, 6))
# # plt.plot(eigenvalues)
# # plt.xlabel('Stocks')
# # plt.ylabel('Maximum Icon Vector Value')
# # plt.title('Maximum Icon Vector Value for Each Stock')
# # plt.grid(axis='y')
# # plt.tight_layout()
# # plt.show()
#==============================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date")

# Calculate log returns
log_returns = np.log(df / df.shift(1))

# Calculate the correlation matrix
correlation_matrix = log_returns.corr()

# Calculate the distance matrix using the formula D_ij = sqrt(2(1 - C_ij))
distance_matrix = np.sqrt(2 * (1 - correlation_matrix))

# Compute the eigenvalues and eigenvectors of the distance matrix
# eigenvalues, eigenvectors = np.linalg.eig(np.abs(correlation_matrix))
# print(eigenvalues)
eigenValues, eigenVectors = np.linalg.eig(distance_matrix)
idx = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
print(eigenValues)
# plt.plot(eigenValues)
# plt.show()
# # Determine which eigenvalue (first or last) is higher
# first_eigenvalue_higher = eigenvalues[0] > eigenvalues[-1]
#
# # Create DataFrame for eigenvectors
# eigenvectors_df = pd.DataFrame(eigenvectors, index=correlation_matrix.index, columns=[f'Eigenvector {i+1}' for i in range(eigenvectors.shape[1])])
#
# # Plot the eigenvectors corresponding to the higher eigenvalue (first or last) for all stocks
# plt.figure(figsize=(12, 8))
# if first_eigenvalue_higher:
#     plt.plot(eigenvectors_df.index, eigenvectors_df.iloc[:, 0], 'b-')
# else:
#     plt.plot(eigenvectors_df.index, eigenvectors_df.iloc[:, -1], 'b-')
#
# plt.title('Eigenvectors Corresponding to the Higher Eigenvalue (First or Last) for All Stocks')
# plt.xlabel('Stocks')
# plt.ylabel('Eigenvector Values')
# plt.grid(True)
# plt.show()
#
# # Save the eigenvectors to a CSV file
# eigenvectors_df.to_csv('icon_vectors.csv')
#
# # Save the eigenvalues to a CSV file
# eigenvalues_df = pd.DataFrame(eigenvalues, index=[f'Eigenvalue {i+1}' for i in range(len(eigenvalues))])
# eigenvalues_df.to_csv('eigenvalues.csv', header=False)
#
# print('Icon vectors have been saved to icon_vectors.csv')
# print('Eigenvalues have been saved to eigenvalues.csv')
#
# # Print which eigenvalue was higher
# higher_eigenvalue = 'First' if first_eigenvalue_higher else 'Last'
# print(f'The higher eigenvalue is the {higher_eigenvalue} eigenvalue.')



