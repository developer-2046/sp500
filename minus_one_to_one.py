import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('SP500_25May24_cleaned_filled.csv', index_col="Date")

# Parse the sector information from the column names
sectors = {col.split('_')[0]: col.split('_')[1] for col in df.columns}
df.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], sectors[col.split('_')[0]]) for col in df.columns], names=['Symbol', 'Sector'])

# Sort columns by sector
df = df.sort_index(axis=1, level=1)

# Calculate log returns
log_returns = np.log(df / df.shift(1))

# Drop NaN values resulting from the shift
log_returns = log_returns.dropna()

# Calculate the correlation matrix
correlation_matrix = log_returns.corr()

# Check the range of correlation matrix values
print(f"Correlation matrix values range from {correlation_matrix.min().min()} to {correlation_matrix.max().max()}")

# Get unique sector codes for labeling
unique_sectors = sorted(set(sectors.values()))

# Create a mapping from sector code to index for labeling
sector_to_index = {sector: i for i, sector in enumerate(unique_sectors)}

# Create a sector label array for the correlation matrix
sector_labels = [sectors[symbol] for symbol in correlation_matrix.columns.get_level_values('Symbol')]

# Plot the correlation matrix
plt.figure(figsize=(14, 12))
plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect='auto')
plt.colorbar()

# Set the tick labels to the sector codes using the sector_to_index mapping
plt.xticks(range(len(sector_labels)), [sector_to_index[label] for label in sector_labels], rotation=90, fontsize=8)
plt.yticks(range(len(sector_labels)), [sector_to_index[label] for label in sector_labels], fontsize=8)

plt.title("Correlation Matrix of S&P 500 Stocks by GICS Sector", fontsize=16)
plt.show()

print(correlation_matrix.describe())
