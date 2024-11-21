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
eigenvalues, eigenvectors = np.linalg.eig(distance_matrix)

# Create a DataFrame for eigenvectors
eigenvectors_df = pd.DataFrame(eigenvectors, index=correlation_matrix.index, columns=[f'Eigenvector {i+1}' for i in range(eigenvectors.shape[1])])

# Save the eigenvectors to a CSV file
eigenvectors_df.to_csv('eigenvectors.csv')

# Save the eigenvalues to a CSV file
#eigenvalues_df = pd.DataFrame(eigenvalues, index=[f'Eigenvalue {i+1}' for i in range(len(eigenvalues))])
eigenvalues_df = pd.DataFrame(eigenvalues)

eigenvalues_df.to_csv('eigenvalues2.csv', header=False)

print('Icon vectors have been saved to eigenvectors.csv')
print('Eigenvalues have been saved to eigenvalues.csv')
