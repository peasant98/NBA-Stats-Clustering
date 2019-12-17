# hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
from clustering.Cluster import NBACluster
import numpy as np

class NBAHierarchical(NBACluster):
    def fit(self, dist_metric, linkage='ward'):
        self.method = 'Hierarchical'
        if linkage == 'ward':
            print('Ward linkage selected. Euclidean metric selected.')
            dist_metric = 'euclidean'
        self.engine = AgglomerativeClustering(n_clusters=self.num_clusters, affinity=dist_metric,
                                                linkage=linkage)
        # fit the data
        cluster = self.engine.fit(self.x)
        self.labels = cluster.labels_
        self.children = cluster.children_
