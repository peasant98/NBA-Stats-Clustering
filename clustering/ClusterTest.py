import numpy as np
from sklearn.cluster import AgglomerativeClustering

X = np.array([[1, 2], [1, 4], [1, 0],
               [4, 2], [4, 4], [4, 0]])
clustering = AgglomerativeClustering().fit(X)

print(clustering)