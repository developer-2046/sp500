# import pandas as pd
# import numpy as np
#
# # Load the distance matrix
# distance_matrix = pd.read_csv('distance_matrix.csv')
#
# # Calculate lambda max directly from each row (interpreted as a vector)
# def calculate_lambda_max(vector):
#     eigenvalues = np.linalg.eigvals(np.outer(vector, vector))  # Example operation
#     return np.max(np.real(eigenvalues))
#
# lambda_max_values = [calculate_lambda_max(row) for row in distance_matrix.values]
#
# # Create a DataFrame for the results
# distance_lambda_df = pd.DataFrame({
#     'time': np.arange(len(lambda_max_values)),
#     'lambda_max': lambda_max_values
# })
#
# # Save the DataFrame to CSV
# distance_lambda_df.to_csv('distance_lambda.csv', index=False)
#
# print("distance_lambda.csv has been created successfully.")
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file containing time, distance, and lambda max
df = pd.read_csv('distance_lambda.csv')  # Replace with your actual CSV file path

# Display the first few rows to check the structure
print(df.head())

# Step 2: Plot the time series for lambda max
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['lambda_max'], marker='o')  # Replace 'time' with appropriate index if necessary
plt.title('Time Series for Lambda Max')
plt.xlabel('Time')
plt.ylabel('Lambda Max')
plt.grid(True)
plt.show()


