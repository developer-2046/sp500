import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kde

# Load your SP500 data and calculate returns
df = pd.read_csv('SP500_25May24_cleaned_filled.csv', index_col=0)
df = df.pct_change().dropna()  # Calculate daily returns

# Step 1: Calculate the covariance matrix
cov_matrix = np.cov(df.T)

# Step 2: Calculate the eigenvalues of the covariance matrix
eigenvalues, _ = np.linalg.eigh(cov_matrix)

# Step 3: Define parameters for the Marchenko-Pastur distribution
T = len(df)  # Number of observations (e.g., days)
N = df.shape[1]  # Number of assets (e.g., stocks)
Q = T / N  # Aspect ratio
sigma_squared = 1  # Assumes returns are standardized

# Calculate the theoretical bounds of the Marchenko-Pastur distribution
lambda_min = sigma_squared * (1 - np.sqrt(1 / Q)) ** 2
lambda_max = sigma_squared * (1 + np.sqrt(1 / Q)) ** 2

# Step 4: Generate Marchenko-Pastur distribution
def marchenko_pastur_pdf(lmbda, Q, sigma_squared=1):
    return (Q / (2 * np.pi * sigma_squared * lmbda)) * np.sqrt((lambda_max - lmbda) * (lmbda - lambda_min))

# Generate eigenvalues within the bounds for the Marchenko-Pastur distribution
lmbda_vals = np.linspace(lambda_min, lambda_max, 1000)
mp_pdf_vals = marchenko_pastur_pdf(lmbda_vals, Q)

# Step 5: Plot empirical eigenvalues and Marchenko-Pastur distribution
plt.figure(figsize=(10, 6))

# Plot histogram of empirical eigenvalues with adjusted range
plt.hist(eigenvalues, bins=50, density=True, alpha=0.6, color='skyblue', label='Empirical eigenvalues')

# Plot the Marchenko-Pastur theoretical distribution
plt.plot(lmbda_vals, mp_pdf_vals, color='red', label='Marchenko-Pastur distribution')

# Adjust y-axis limits (set manually)
plt.ylim(0, max(mp_pdf_vals) * 1.2)  # Adjust this multiplier if necessary

plt.xlabel('Eigenvalue')
plt.ylabel('Density')
plt.title('Empirical Eigenvalue Distribution vs. Marchenko-Pastur')
plt.legend()
plt.grid(True)
plt.show()
