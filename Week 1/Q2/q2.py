import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA
    print(init_array)
    mean_arr = np.mean(init_array)
    print(mean_arr)

    # 1. standardized Array (not dividing by std dev)
    stdz_array = init_array - mean_arr
    print(stdz_array)
    
    # 2. Covariance matrix (means of the standardised dataset are 0)
    Cov_matrix = stdz_array.cov()

    # 3. Calculate the eigenvalues and eigenvectors for the covariance matrix.
    eigenvalues, eigenvectors = np.linalg.eig(Cov_matrix)
    print(eigenvalues)

    # 4. Sort the eigenvalues in decreasing order
    real_indices = np.isreal(eigenvalues)
    # complex_indices = np.isreal(eigenvalues)
    real_eigenvalues = eigenvalues[real_indices]
    # complex_eigenvalues = eigenvalues[complex_indices]
    sorted_indices = np.argsort(real_eigenvalues)
    sorted_eigenvalues = np.sort(real_eigenvalues)
    sorted_eigenvectors = eigenvectors[:, real_indices][:, sorted_indices]
    sorted_eigenvectors_df = pd.DataFrame(sorted_eigenvectors)

    final_data = init_array.dot(sorted_eigenvectors_df)
    # END TODO

    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)

    # TODO: plot and save a scatter plot of final_data to out.png
    plt.scatter(final_data[0]+final_data[1], init_array[0]+init_array[1] , marker='o')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.savefig('out.png')
    plt.show()
    # END TODO
