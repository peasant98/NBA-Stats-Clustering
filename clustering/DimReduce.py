# various techniques for dimension reduction
import numpy as np
import pandas as pd

# pca from scratch
def pca(m, dims):
    k = min(m.shape[-1], dims)
    m_t = m.T
    mm_t = np.matmul(m, m_t)
    m_tm = np.matmul(m_t, m)
    matrix = mm_t if mm_t.shape[0] < m_tm.shape[0] else m_tm

    eigvals, eigvecs = np.linalg.eig(matrix)
    e = np.array([x for _,x in sorted(zip(eigvals,eigvecs))[::-1]])
    ek = e[:,:k]
    me2 = np.matmul(m, ek)
    # return dimension reduced matrix
    columns = [f'x{i}' for i in range(me2.shape[-1])]
    
    df = pd.DataFrame(me2, columns=columns)
    res = df[columns].values
    print(res)
    return res, df, columns
