# NBA Stats Clustering
# Copyright Matthew Strong, 2019

# gaussian mixture models with em algorithm
import numpy as np
from scipy import stats
from clustering.Cluster import NBACluster

# nba gmm class
# gmm from scratch as well, more explained below
class NBAGMM(NBACluster):
    def fit(self):
        self.method = 'GMM'
        # get the points
        a, m = self.get_points(self.num_clusters)
        # em algorithm for 100 iterations
        res = self.em_algorithm(self.num_clusters, m, a)
        probs_given_data = res[2]
        # probability of each point
        # sum of squared disatnces
        # and get assignments by max probability of each point to a certain cluster
        l = []
        dist = 0
        for v in range(len(a)):
            selection = np.argmax(probs_given_data[:,v])
            dist += self.dist(a[v], res[0][selection])
            l.append(selection)
        self.ssd = dist
        self.labels = l
        self.centroids = res[0]

    def get_points(self, k):
        # select points randomly
        a = self.df.values
        indices = np.random.choice(list(range(len(a))), k, replace=False)
        k_points = a[indices]
        return a, k_points

    def dist(self, x1, x2):
        # euclidean distance
        return np.sqrt(np.sum((x1-x2)**2))

    # this algorithm was influenced by the gmm in class notebook, as well as my implementation in hw2
    # but heavily adapted for n dimensions and varying values of k, now all vectorized, so, more
    # dynamic to work with.
    def em_algorithm(self, k, m, a):
        # works for n dimensional data
        # pick k random points
        # get means from the randomly selected data
        # works
        mu = np.zeros((k, a.shape[-1]))
        covariances = np.zeros((k, a.shape[-1], a.shape[-1]))
        probs = np.zeros(k)
        # also p_class_n
        # set probabilities of each cluster, or the weight, to all equal
        probs.fill(1./k)
        # calculations of prob of m give data require these matrices
        p_given_class = np.zeros((k, len(a)))
        p_given_data = np.zeros((k, len(a)))
        p_class_data = np.zeros((k, len(a), 1, 1))
        n_class = np.zeros(k)
        for ind,val in enumerate(mu):
            mu[ind] = m[ind]
        for ind,val in enumerate(covariances):
            # set all covariances of k mixtures to overall covariance of dataset
            if ind == 0:
                covariances[0] = np.cov(a.T)
            else:
                covariances[ind] = covariances[0]
        for _ in range(100):
            # 100 iterations
            summation = np.zeros((len(a)))
            for i in range(k):
                # compute pdf
                p_given_class[i] = stats.multivariate_normal.pdf(a, mean=mu[i], cov=covariances[i], allow_singular=True)
                p_given_data[i] = p_given_class[i] * probs[i]
                summation += p_given_data[i]
            length = len(a)
            for i in range(k):
                # get probabilities of mixtures
                p_given_data[i]/=summation
                n_class[i] = np.sum(p_given_data[i])
                probs[i] = n_class[i]/length
            for i in range(k):
                means = np.zeros(a.shape[-1])
                # get means from data
                for j in range(len(means)):
                    means[j] = (1.0/n_class[i]) * np.sum(p_given_data[i]*a[:,j])
                mu[i] = np.array(means)
            for i in range(k):
                # covariance calculations
                covs = []
                for p in a:
                    x_i = p
                    r = x_i - mu[i]
                    vec = np.expand_dims(r, axis=0)
                    cov_i = vec * vec.T
                    covs.append(cov_i)
                # expand dims and use np sum to get results along axis=0
                covs = np.array(covs)
                temp = np.expand_dims(p_given_data[i], axis=1)
                p_class_data[i] = np.expand_dims(temp, axis=1)
                covariances[i] = np.sum(p_class_data[i] * covs, axis=0) / n_class[i]
        # return means, covariances of cluster, probabilities of points being in certain mixture
        # and probabilities of mixtures themselves.
        return mu, covariances, p_given_data, probs
