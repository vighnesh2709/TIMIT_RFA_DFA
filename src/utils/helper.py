import numpy as np
from sklearn.decomposition import PCA
import torch

def file_writer(name, data):
    with open(f"logs/{name}.txt", "w") as f:
        for value in data:
            f.write(f"{value}\n")


def get_orthogonal_matrix(dim_1, dim_2):

    X = np.random.randn(dim_1 * 3, dim_2)
    pca = PCA(n_components=dim_1 * 2)
    pca.fit(X)

    matrix = pca.components_
    print("SHAPE OF THE MATRIX BEFORE SPLITTING IS")
    print(matrix.shape)

    # Split Top and Bottom Half
    # matrix_1 = matrix[:dim_1]
    # matrix_2 = matrix[dim_1:]

    # Alternative split
    matrix_1 = matrix[::2][:dim_1]
    matrix_2 = matrix[1::2][:dim_1]

    print("MATRIX 1 SHAPE:", matrix_1.shape)
    print("MATRIX 2 SHAPE: ", matrix_2.shape)
    print("ORTHOGONALITY CHECK: ", np.trace(matrix_1.T @ matrix_2))
    print("ORTHOGONALITY CHECK: ", np.sum(matrix_1 * matrix_2))

    return matrix_1, matrix_2

def orthogonal_pair(dim=1024):
    A = np.random.randn(dim, dim)

    B = np.random.randn(dim, dim)

    # flatten
    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)

    # remove projection of B onto A
    B_flat = B_flat - (B_flat @ A_flat) / (A_flat @ A_flat) * A_flat

    B = B_flat.reshape(dim, dim)
    
    print("MATRIX 1 SHAPE:", A.shape)
    print("MATRIX 2 SHAPE: ", B.shape)
    print("ORTHOGONALITY CHECK: ", np.trace(B.T @ A))
    print("ORTHOGONALITY CHECK: ", np.sum(B * A))
    return B, A

def data_pca(dim_1, dim_2):
    data = torch.load("/speech/malar/vighnesh/timit_rfa_dfa/data/processed_13/X.pt")
    print(type(data))
    print(data.shape)
    
    data = data.T
    print(data.shape)
    new_data = data[:dim_1 * 3, : dim_2]
    pca = PCA(n_components = dim_1 * 2)
    pca.fit(new_data)


    matrix = pca.components_
    print("SHAPE OF THE MATRIX BEFORE SPLITTING IS")
    print(matrix.shape)

    # Alternative split
    matrix_1 = matrix[::2][:dim_1]
    matrix_2 = matrix[1::2][:dim_1]

    print("MATRIX 1 SHAPE:", matrix_1.shape)
    print("MATRIX 2 SHAPE: ", matrix_2.shape)
    print("ORTHOGONALITY CHECK: ", np.trace(matrix_1.T @ matrix_2))
    print("ORTHOGONALITY CHECK: ", np.sum(matrix_1 * matrix_2))

    return matrix_1, matrix_2



#orthogonal_pair()
#get_orthogonal_matrix(48,1024)
# data_pca(48,1024)    
