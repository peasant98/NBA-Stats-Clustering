# hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
class NBAHierarchical(NBACluster):
    pass


import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
               [4, 2], [4, 4], [4, 0]])
clustering = AgglomerativeClustering().fit(X)
print(clustering)
# def __init__(self, type_clustering, dim_reduce=False):
#         pass

#     def init_data(self, data):
#         pass

#     def fit(self, eps):
#         # fit the data
#         pass
#     def fit(self, test_data):
#         # test the data
#         pass