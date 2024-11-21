# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# #
# # # Load the data
# # df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date")
# #
# # # Calculate log returns
# # log_returns = np.log(df / df.shift(1))
# #
# # # Calculate the correlation matrix
# # correlation_matrix = log_returns.corr()
# #
# # # Calculate the distance matrix using the formula D_ij = sqrt(2(1 - C_ij))
# # distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
# #
# # distance_df = pd.DataFrame(distance_matrix)
# # distance_df.to_csv('distance_matrix.csv', index=False, header=False)
# #
# # # Plot the distance matrix as a heatmap
# # plt.figure(figsize=(10, 8))
# # sns.heatmap(distance_matrix, cmap='viridis', cbar_kws={'label': 'Distance'})
# # plt.title('Distance Matrix')
# # plt.xlabel('Stocks')
# # plt.ylabel('Stocks')
# #
# # # Remove the names of stocks from x and y axes
# # plt.xticks([], [])
# # plt.yticks([], [])
# #
# # plt.show()
# # -------------------------------------------------------------------------
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load the data
# df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date")
#
# # Calculate log returns
# log_returns = np.log(df / df.shift(1))
#
# # Calculate the correlation matrix
# correlation_matrix = log_returns.corr()
#
# # Calculate the distance matrix using the formula D_ij = sqrt(2(1 - C_ij))
# distance_matrix = np.sqrt(2 * (1 - correlation_matrix))
#
# # Convert to DataFrame to ensure row/column names are included
# distance_df = pd.DataFrame(distance_matrix, index=correlation_matrix.index, columns=correlation_matrix.columns)
#
# # Ensure the matrix is square
# if distance_df.shape[0] == distance_df.shape[1]:
#     print(f"Matrix is square: {distance_df.shape}, saving correctly.")
#     # Save with index and header
#     distance_df.to_csv('distance_matrix_final.csv')
# else:
#     print(f"Matrix is not square: {distance_df.shape}. Check your calculations!")
#
# # Plot the distance matrix as a heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(distance_matrix, cmap='viridis', cbar_kws={'label': 'Distance'})
# plt.title('Distance Matrix')
# plt.xlabel('Stocks')
# plt.ylabel('Stocks')
#
# # Remove the names of stocks from x and y axes
# plt.xticks([], [])
# plt.yticks([], [])
#
# plt.show()
#
#-------------------------------------------------------------------
import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv('../SP500_25May24_cleaned_filled.csv', index_col="Date")

# Fill missing values, if any, using forward fill or any other method
df.fillna(method='ffill', inplace=True)

# Calculate log returns (handling NaNs after the shift)
log_returns = np.log(df / df.shift(1))

# Drop any rows with NaN values after calculating returns
log_returns.dropna(inplace=True)

# Calculate the correlation matrix
correlation_matrix = log_returns.corr()

# Verify if the correlation matrix is square
if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
    print(f"Correlation matrix is not square: {correlation_matrix.shape}. Check data processing.")
else:
    print(f"Correlation matrix is square: {correlation_matrix.shape}.")

# Calculate the distance matrix using the formula D_ij = sqrt(2(1 - C_ij))
distance_matrix = np.sqrt(2 * (1 - correlation_matrix))

# Convert to DataFrame with proper indices and columns
distance_df = pd.DataFrame(distance_matrix, index=correlation_matrix.index, columns=correlation_matrix.columns)

# Verify that the matrix is square before saving
if distance_df.shape[0] != distance_df.shape[1]:
    print(f"Matrix is not square: {distance_df.shape}.")
else:
    print(f"Matrix is square: {distance_df.shape}.")

# Save the distance matrix with indices and headers
distance_df.to_csv('distance_matrix_final2.csv')
