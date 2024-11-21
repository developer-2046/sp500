import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date")

# Parse the sector information from the column names
sectors = {col.split('_')[0]: col.split('_')[1] for col in df.columns}
df.columns = pd.MultiIndex.from_tuples([(col.split('_')[0], sectors[col.split('_')[0]]) for col in df.columns], names=['Symbol', 'Sector'])

# Sort columns by sector
df = df.sort_index(axis=1, level=1)

# Calculate log returns
log_returns = np.log(df / df.shift(1))

# Define a function to plot the correlation matrix
def plot_correlation_matrix(correlation_matrix, sector_labels, sector_to_index, title):
    plt.figure(figsize=(14, 12))
    plt.imshow(correlation_matrix, cmap="coolwarm", aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(sector_labels)), [sector_to_index[label] for label in sector_labels], rotation=90, fontsize=8)
    plt.yticks(range(len(sector_labels)), [sector_to_index[label] for label in sector_labels], fontsize=8)
    plt.title(title, fontsize=16)
    plt.show()

# Get unique sector codes for labeling
unique_sectors = sorted(set(sectors.values()))

# Create a mapping from sector code to index for labeling
sector_to_index = {sector: i for i, sector in enumerate(unique_sectors)}

# Create a sector label array for the correlation matrix
sector_labels = [sectors[symbol] for symbol in log_returns.columns.get_level_values('Symbol')]

# Number of days per window
window_size = 20

# Iterate over the data in windows of `window_size`
for start in range(0, len(log_returns) - window_size + 1, window_size):
    end = start + window_size
    window_data = log_returns.iloc[start:end]
    correlation_matrix = window_data.corr()
    last_date = log_returns.index[end - 1]  # Get the last date of the current window
    title = f"Correlation Matrix up to {last_date}"
    plot_correlation_matrix(correlation_matrix, sector_labels, sector_to_index, title)
