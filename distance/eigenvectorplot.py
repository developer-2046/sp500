import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

distance_matrix = pd.read_csv('distance_matrix_final2.csv', index_col=0)

if 'Unnamed: 0' in distance_matrix.columns:
    distance_matrix.drop(columns=['Unnamed: 0'], inplace=True)

distance_matrix = distance_matrix.apply(pd.to_numeric, errors='coerce')

distance_matrix.dropna(axis=1, inplace=True)  # Drop columns with NaN
distance_matrix.dropna(axis=0, inplace=True)

eigenvalues, eigenvectors = np.linalg.eig(distance_matrix.values)

# Load the eigenvectors from 'eigenvectors.csv'
eigenvectors_df = pd.read_csv('eigenvectors.csv')

# Display the DataFrame to check its contents
print("DataFrame loaded from 'eigenvectors.csv':")

# Drop non-numeric columns like 'Unnamed: 0' to focus on eigenvectors
eigenvectors_df = eigenvectors_df
plt.figure(figsize=(10, 6))
plt.plot(eigenvectors_df['name'], eigenvectors_df['Eigenvector 1'])
plt.grid(False)
plt.show()
