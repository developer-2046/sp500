import numpy as np
import pandas as pd
import scipy.stats as st

def calc(prob_vector):
    entropy = 0
    for p in prob_vector:
        if p > 0:  # Shannon entropy is undefined for p=0
            entropy -= p * np.log(np.sqrt(p))  # Natural log used here
    return entropy

def entropy_from_distance_matrix(csv_file):

    # Load the distance matrix from the CSV file
    distance_matrix = pd.read_csv(csv_file, index_col=0)

    if 'Unnamed: 0' in distance_matrix:
        distance_matrix.drop(columns=['Unnamed: 0'], inplace=True)

    # Convert all values to numeric, coercing errors to NaN and dropping any rows/columns with non-numeric values
    distance_matrix = distance_matrix.apply(pd.to_numeric, errors='coerce')

    # Drop any NaN values (non-numeric data that couldn't be converted)
    distance_matrix.dropna(axis=1, inplace=True)
    distance_matrix.dropna(axis=0, inplace=True)

    # Flatten the matrix into a 1D vector
    flat_vector = np.abs(distance_matrix.values).flatten()
    flattened_matrix = flat_vector[~np.isnan(flat_vector)]
    flattened_matrix = flattened_matrix[flattened_matrix > 0]
    print(flat_vector)
    print(np.sum(flattened_matrix))
    flattened_matrix = np.abs(flattened_matrix)
    prob_vector = flattened_matrix / np.sum(flattened_matrix)

    # Calculate Shannon entropy using the probability vector
    print(np.sum(prob_vector))
    print(len(prob_vector))
    entropy = calc(prob_vector)

    return entropy

# Example usage:
csv_file = '../RMTpaper/distance_matrix_final2.csv'  # Replace with your file path
entropy = entropy_from_distance_matrix(csv_file)
print(f'Shannon Entropy: {entropy}')
