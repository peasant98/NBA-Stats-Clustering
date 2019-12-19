# NBA Stats Clustering
# Copyright Matthew Strong, 2019
from sklearn.cluster import AgglomerativeClustering
from clustering.Cluster import NBACluster
import numpy as np

# used sklearn hierarchical clustering because of ward linkage (we didn't go over in class)
# effective because it minimizes increased variance when we combine any two clusters.
class NBAHierarchical(NBACluster):
    # fit the data
    def fit(self, dist_metric, linkage='ward'):
        self.method = 'Hierarchical'
        # ward linkage computes variance if two clusters are connected
        # uses euclidean distance as dist_metric
        if linkage == 'ward':
            print('Ward linkage selected. Euclidean metric selected.')
            dist_metric = 'euclidean'
        # I implemented sklearn AgglomerativeClustering here, good job they did 
        self.engine = AgglomerativeClustering(n_clusters=self.num_clusters, affinity=dist_metric,
                                                linkage=linkage)
        # perform clustering
        # fit the data
        # get the means from the clusters, put in centroids list
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
        # sum of squared distances from each point to its centroid
        dist = 0
        for v in range(len(self.x)):
            dist += self.dist(self.x[v], centroids[labels_np[v]])
        self.ssd = dist
        self.centroids = np.array(centroids)
        self.children = cluster.children_
    
    def dist(self, x1, x2):
        # distance, euclidean
        return np.sqrt(np.sum((x1-x2)**2))
