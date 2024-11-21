import pandas as pd
import matplotlib.pyplot as plt

# Load the eigenvectors.csv file
eigenvectors_df = pd.read_csv('eigenvectors.csv')

# Assuming each column represents an eigenvector and the file contains numeric data
# Select the 35th eigenvector (since index starts at 0, it's index 34)
eigenvector_35 = eigenvectors_df.iloc[:, 34]

# Plot the 35th eigenvector
plt.figure(figsize=(10, 6))
plt.plot(eigenvector_35, marker='o')
plt.title('Eigenvector 35')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
