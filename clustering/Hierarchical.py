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
        means = np.zeros((self.num_clusters, self.x.shape[-1]))
        cluster = self.engine.fit(self.x)
        self.labels = cluster.labels_
        labels_np = np.array(self.labels)
        centroids = []
        for val in range(self.num_clusters):
            indices = np.where(labels_np==val)
            mean = np.mean(self.x[indices], axis=0)
            centroids.append(mean)
            # get indices of the corresponding label
        dist = 0
        for v in range(len(self.x)):
            dist += self.dist(self.x[v], centroids[labels_np[v]])
        self.ssd = dist

        self.children = cluster.children_
    
    def dist(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
