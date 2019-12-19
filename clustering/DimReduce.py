# NBA Stats Clustering
# Copyright Matthew Strong, 2019
# various techniques for dimension reduction
import numpy as np
import pandas as pd

# pca from scratch
# influenced by in-class notebook I did on pca.
def pca(m, dims):
    # k dims
    k = min(m.shape[-1], dims)
    m_t = m.T
    mm_t = np.matmul(m, m_t)
    m_tm = np.matmul(m_t, m)
    # select matrix based on square matrix with smaller dimensions
    matrix = mm_t if mm_t.shape[0] < m_tm.shape[0] else m_tm

    # compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(matrix)
    # get sorted eigenvalues, and we'll use the k principal eigenvectors
    e = np.array([x for _,x in sorted(zip(eigvals,eigvecs))[::-1]])
    # get those eigenvectors
    ek = e[:,:k]
    # multiply by original data to get reduced dimension data
    me2 = np.matmul(m, ek)
    columns = [f'x{i}' for i in range(me2.shape[-1])]
    # put into datafram
    df = pd.DataFrame(me2, columns=columns)
    res = df[columns].values
    # get values from reduced columns
    print(res)
    # return result
    return res, df, columns
