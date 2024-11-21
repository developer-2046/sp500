# principle component analysis
""" apply Principal Component Analysis (PCA) to the eigenvector matrix to
reduce dimensionality and find meaningful patterns. This is particularly useful
for exploring correlations between different stocks."""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

distance_matrix = pd.read_csv('distance_matrix_final2.csv', index_col=0)

if 'Unnamed: 0' in distance_matrix.columns:
    distance_matrix.drop(columns=['Unnamed: 0'], inplace=True)

distance_matrix = distance_matrix.apply(pd.to_numeric, errors='coerce')

distance_matrix.dropna(axis=1, inplace=True)  # Drop columns with NaN
distance_matrix.dropna(axis=0, inplace=True)

eigenvalues, eigenvectors = np.linalg.eig(distance_matrix.values)

# Load the eigenvectors from 'eigenvectors.csv'
eigenvector_matrix = np.array(eigenvectors)

pca = PCA(n_components=5)
pca.fit(eigenvector_matrix)

# Explained variance tells how much each principal component accounts for variance in the data
explained_variance = pca.explained_variance_ratio_
plt.bar(range(1, 6), explained_variance)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA on Eigenvectors')
plt.show()

projected_data = pca.transform(eigenvector_matrix)
plt.scatter(projected_data[:, 0], projected_data[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA Projection of Eigenvectors')
plt.show()