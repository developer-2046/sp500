import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Load SP500 data and calculate returns
df = pd.read_csv('SP500_25May24_cleaned_filled.csv', index_col=0)
df = df.pct_change().dropna()  # Calculate daily returns

# Set rolling window size
window_size = 20

# Initialize list to store eigenvalues
eigenvalues = []

# Compute Wishart matrices for each rolling window
for start in range(len(df) - window_size + 1):
    # Get data for the current window
    window_data = df.iloc[start:start + window_size]

    # Calculate the pairwise distance matrix
    distance_matrix = squareform(pdist(window_data.T, metric='euclidean'))

    # Center the distance matrix
    centered_distance_matrix = distance_matrix - np.mean(distance_matrix, axis=0)

    # Compute the Wishart matrix
    wishart_matrix = centered_distance_matrix.T @ centered_distance_matrix

    # Compute eigenvalues of the Wishart matrix
    eigvals = np.linalg.eigvalsh(wishart_matrix)

    # Append normalized eigenvalues
    eigenvalues.extend(eigvals / df.shape[1])  # Normalize by the number of assets (N)

# Convert eigenvalues to a NumPy array
eigenvalues = np.array(eigenvalues)

# Define parameters for the Marchenko-Pastur distribution
N = df.shape[1]  # Number of stocks (assets)
T = window_size  # Window size (number of observations)
Q = T / N  # Aspect ratio
lambda_min = (1 - np.sqrt(1 / Q)) ** 2
lambda_max = (1 + np.sqrt(1 / Q)) ** 2

# Generate the Marchenko-Pastur distribution
lmbda_vals = np.linspace(lambda_min, lambda_max, 1000)
mp_pdf = lambda lmbda: Q / (2 * np.pi * lmbda) * np.sqrt((lambda_max - lmbda) * (lmbda - lambda_min))
mp_pdf_vals = mp_pdf(lmbda_vals)

# Plot the empirical eigenvalue distribution and the theoretical Marchenko-Pastur distribution
plt.figure(figsize=(10, 6))
plt.hist(eigenvalues, bins=100, density=True, alpha=0.6, color='skyblue', label='Empirical eigenvalues')
plt.plot(lmbda_vals, mp_pdf_vals, color='red', label='Marchenko-Pastur distribution')

# Adjust y-axis limits (manually or log-scale)
plt.ylim(0, max(mp_pdf_vals) * 1.2)  # Adjust this multiplier if necessary
# plt.yscale('log')  # Uncomment to use a logarithmic scale

plt.xlabel('Normalized Eigenvalue')
plt.ylabel('Density')
plt.title('Eigenvalue Distribution of Wishart Ensemble')
plt.legend()
plt.grid(True)
plt.show()
