# gaussian mixture models with em algorithm
import numpy as np
from scipy import stats
from clustering.Cluster import NBACluster

class NBAGMM(NBACluster):
    def fit(self):
        self.method = 'GMM'
        a, m = self.get_points(self.num_clusters)
        res = self.em_algorithm(self.num_clusters, m, a)
        probs_given_data = res[2]
        l = []
        for v in range(len(a)):
            l.append(np.argmax(probs_given_data[:,v]))
        self.labels = l
    def get_points(self, k):
        a = self.df.values
        indices = np.random.choice(list(range(len(a))), k, replace=False)
        k_points = a[indices]
        return a, k_points

    def em_algorithm(self, k, m, a):
        # pick k random points
        mu = np.zeros((k, a.shape[-1]))
        covariances = np.zeros((k, a.shape[-1], a.shape[-1]))
        probs = np.zeros(k)
        # also p_class_n
        probs.fill(1./k)
        p_given_class = np.zeros((k, len(a)))
        p_given_data = np.zeros((k, len(a)))
        p_class_data = np.zeros((k, len(a), 1, 1))
        n_class = np.zeros(k)
        for ind,val in enumerate(mu):
            mu[ind] = m[ind]
        for ind,val in enumerate(covariances):
            if ind == 0:
                covariances[0] = np.cov(a.T)
            else:
                covariances[ind] = covariances[0]
        for _ in range(100):
            summation = np.zeros((len(a)))
            for i in range(k):
                p_given_class[i] = stats.multivariate_normal.pdf(a, mean=mu[i], cov=covariances[i], allow_singular=True)
                p_given_data[i] = p_given_class[i] * probs[i]
                summation += p_given_data[i]
            length = len(a)
            for i in range(k):
                p_given_data[i]/=summation
                n_class[i] = np.sum(p_given_data[i])
                probs[i] = n_class[i]/length
            for i in range(k):
                means = np.zeros(a.shape[-1])
                for j in range(len(means)):
                    means[j] = (1.0/n_class[i]) * np.sum(p_given_data[i]*a[:,j])
                mu[i] = np.array(means)
            for i in range(k):
                covs = []
                for p in a:
                    x_i = p
                    r = x_i - mu[i]
                    vec = np.expand_dims(r, axis=0)
                    cov_i = vec * vec.T
                    covs.append(cov_i)
                covs = np.array(covs)
                temp = np.expand_dims(p_given_data[i], axis=1)
                p_class_data[i] = np.expand_dims(temp, axis=1)
                covariances[i] = np.sum(p_class_data[i] * covs, axis=0) / n_class[i]
        return mu, covariances, p_given_data, probs
