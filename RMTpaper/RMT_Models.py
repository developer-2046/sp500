from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from RMTpaper.PCA import distance_matrix


def marchenko_pastur_distribution(x, q):
    return (1 / (2 * np.pi * x * q)) * np.sqrt((q * (1 + np.sqrt(1 / q))**2 - x) * (x - q * (1 - np.sqrt(1 / q))**2))

distance_matrix = pd.read_csv('distance_matrix_final2.csv')
eigenvalues = np.linalg.eigvalsh(distance_matrix)

plt.hist(eigenvalues, bins=50, density=True, alpha=0.6, color='g', label='Empirical')

x = np.linspace(min(eigenvalues), max(eigenvalues), 1000)
plt.plot(x, marchenko_pastur_distribution(x, q=0.5), label='Marchenko-Pastur Fit')
plt.xlabel('Eigenvalue')
plt.ylabel('Density')
plt.title('Eigenvalue Spectrum vs Marchenko-Pastur Distribution')
plt.legend()
plt.show()
