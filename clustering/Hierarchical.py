# hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
from Cluster import NBACluster
import numpy as np

class NBAHierarchical(NBACluster):
    def fit(self, dist_metric, linkage='ward'):
        if linkage == 'ward':
            print('Ward linkage selected. Euclidean metric selected.')
            dist_metric = 'euclidean'
        self.engine = AgglomerativeClustering(n_clusters=self.num_clusters, affinity=dist_metric,
                                                linkage=linkage)
        # fit the data
        cluster = self.engine.fit(self.x)
        self.labels = cluster.labels_
        self.children = cluster.children_

nba = NBAHierarchical(5)
nba.init_data_from_df('2019-20', ['STL', 'BLK', 'REB'], normalize=True)
nba.fit('euclidean')
nba.plot(True)