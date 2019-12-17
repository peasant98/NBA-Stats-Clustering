from sklearn.cluster import KMeans
from clustering.Cluster import NBACluster
import numpy as np

# k-means clustering on n-dimensional data
class NBAKMeans(NBACluster):
    def fit(self, initialization, max_iter, tolerance):
        self.method = 'KM'
        self.engine = KMeans(n_clusters=self.num_clusters, init=initialization,
                                max_iter=max_iter, tol=tolerance)        
        # fit the data
        cluster = self.engine.fit(self.x)
        self.labels = cluster.labels_
        self.centroids = cluster.cluster_centers_
        print(f'KMeans ran for {cluster.n_iter_} iterations with ssd {cluster.inertia_}')
        self.ssd = cluster.inertia_


class NBAKMeansSimple(NBACluster):

    def fit(self, rand_initialization, tolerance):
        self.method = 'KM-Simple-Random' if rand_initialization else 'KM-Simple-Extreme'
        cluster = self.k_means(self.num_clusters, self.df, tolerance, 
                    random=rand_initialization, cols=np.array(self.df.columns.values))
        # print(cluster) 
        self.labels = cluster[0]
        self.centroids = cluster[2]
        dist = 0
        a = np.array(self.df.values)
        for v in range(len(a)):
            dist += self.dist(a[v], self.centroids[self.labels[v]])
        self.ssd = dist
        print(f'The mean of the whole cluster is {cluster[1]}')

    def dist(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    # z, z1 = get_extreme_points(5, dfy, ['views', 'likes'])
    # get extreme points initialization method for kmeans    
    def get_extreme_points(self, k, df, cols):
        # gets the extreme points of the simulation
        X = []
        for c in cols:
            x = df[c].values
            X.append(x)
        X = np.array(X)
        X = X.T
        # get centers
        center = []
        # ye
        for i in range(X.shape[1]):
            x_bar = np.sum(X[:,i]) / len(X)
            center.append(x_bar)
        center = np.array(center)   
        arr = [(self.dist(center, val), val) for val in X]
        # most extreme points in terms of distance from x_bar or mean
        res = sorted(arr, key=lambda tup: tup[0])[::-1][:k]
        # return the inits, and return the data
        return np.array(list(np.array(res)[:,1])), X

    def update_centroids(self, points, assignments, k, feature_num):
    # generic method for updating centroids based on number of features
        # go through each point, 
        eps = 1e-5
        cluster_tuples = [[0 for i in range(feature_num+1)] for j in range(k)]
        for ind,val in enumerate(points):
            for i in range(feature_num):
                cluster_tuples[assignments[ind]][i] += val[i]
            cluster_tuples[assignments[ind]][-1] += 1
        
        centroids = [[x[i]/(x[-1]+eps) for i in range(feature_num)] for x in cluster_tuples]
        return centroids

    def update_points(self, centroids, values):
        # update the points with the centroids
        assignments = []
        for val in values:
            # get the closest centroid
            lowest = np.inf
            centroid_val = 0
            for ind, ctr in enumerate(centroids):
                d = self.dist(val, ctr)
                if d < lowest:
                    lowest = d 
                    centroid_val = ind
            assignments.append(centroid_val)
        return assignments

    def init_centroids(self, k, df, random=True, other_data=None, cols=None):
        centroids = []
        if random:
            values = self.df.values
            indices = np.random.choice(list(range(len(values))), k, replace=False)
            centroids = values[indices]
        else:
            # pick the most extreme values
            centroids, values = self.get_extreme_points(k, df, cols)
        return centroids, values

    def k_means(self, k, df, tolerance, random=True, other_data=None, cols=None, get_avg_distance=True):
        feature_num = len(cols)
        centroids, values = self.init_centroids(k, df, random, other_data=other_data, cols=cols)
        
        
        while True:
            assignments = self.update_points(centroids, values)
            new_centroids = self.update_centroids(values, assignments, k, feature_num)
            distance = self.dist(np.array(centroids), np.array(new_centroids))
            if distance == 0:
                # stopping condition
                break
            centroids = new_centroids
        mean = None
        if get_avg_distance:
            # get the average distances
            d = 0
            for ind,val in enumerate(values):
                d += self.dist(val, centroids[assignments[ind]])
            mean = d / len(values)
        # return the assigned points' clusters, mean, and the actual centroids.
        return assignments, mean, centroids
