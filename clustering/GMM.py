# gaussian mixture models with em algorithm
import numpy as np
import stats

class NBAGMM(NBACluster):
     def fit():
         pass
    # gmm, done from scratch



# i = [df['x'].values]
# j = [df['y'].values]
# a = np.concatenate((i,j)).T
# sorted_points = a[a[:,0].argsort()]

# m1 = sorted_points[0]
# m2 = sorted_points[-1]

# greatest_dist = -np.inf
# m3 = None
# for point in sorted_points:
#     distance = dist(point, m1) + dist(point, m2)
#     if distance > greatest_dist:
#         greatest_dist = distance
#         m3 = point
# cov = np.cov(a.T)
# m1, m2, m3



# # gmm for 3 clusters
# def EM3(dk, m1, m2, m3, a):
#     mu1 = m1
#     mu2 = m2
#     mu3 = m3
#     cov1 = np.cov(a.T)
#     cov2 = cov1
#     cov3 = cov1
#     pi = 0.333
#     p_class1 = pi
#     p_class2 = pi
#     p_class3 = pi
    
#     for _ in range(100):
#         p_data_given_class1 = stats.multivariate_normal.pdf(a, mean=mu1, cov=cov1)
#         p_data_given_class2 = stats.multivariate_normal.pdf(a, mean=mu2, cov=cov2)
#         p_data_given_class3 = stats.multivariate_normal.pdf(a, mean=mu3, cov=cov3)
        
#         p_class1_given_data = p_data_given_class1*p_class1
#         p_class2_given_data = p_data_given_class2*p_class2
#         p_class3_given_data = p_data_given_class3*p_class3
        
#         summ = p_class1_given_data + p_class2_given_data + p_class3_given_data
#         p_class1_given_data = p_class1_given_data / summ
#         p_class2_given_data = p_class2_given_data / summ
#         p_class3_given_data = p_class3_given_data / summ
        
#         n_class1 = np.sum(p_class1_given_data)
#         n_class2 = np.sum(p_class2_given_data)
#         n_class3 = np.sum(p_class3_given_data)
        
#         # important stuff right here
#         length = len(a)
#         p_class1 = n_class1/length
#         p_class2 = n_class2/length
#         p_class3 = n_class3/length
        
#         x_bar = (1/n_class1) * np.sum(p_class1_given_data*a[:,0])
#         y_bar = (1/n_class1) * np.sum(p_class1_given_data*a[:,1])
#         mu1 = np.array([x_bar, y_bar])
        
#         x_bar = (1/n_class2) * np.sum(p_class2_given_data*a[:,0])
#         y_bar = (1/n_class2) * np.sum(p_class2_given_data*a[:,1])
#         mu2 = np.array([x_bar, y_bar])
        
#         x_bar = (1/n_class3) * np.sum(p_class3_given_data*a[:,0])
#         y_bar = (1/n_class3) * np.sum(p_class3_given_data*a[:,1])
#         mu3 = np.array([x_bar, y_bar])
        
#         # get covariances
#         covs = []
#         for p in a:
#             x_i = p
#             r = x_i - mu1
#             vec = np.expand_dims(r, axis=0)
#             cov_i = vec * vec.T
#             covs.append(cov_i)
#         covs = np.array(covs)
        
#         p_class_1_data = np.expand_dims(p_class1_given_data, axis=1)
#         p_class_1_data = np.expand_dims(p_class_1_data, axis=1)
#         cov1 = np.sum(p_class_1_data * covs, axis=0) / n_class1
        
#         covs = []
#         for p in a:
#             x_i = p
#             r = x_i - mu2
#             vec = np.expand_dims(r, axis=0)
#             cov_i = vec * vec.T
#             covs.append(cov_i)
#         covs = np.array(covs)
        
#         p_class_2_data = np.expand_dims(p_class2_given_data, axis=1)
#         p_class_2_data = np.expand_dims(p_class_2_data, axis=1)
#         cov2 = np.sum(p_class_2_data * covs, axis=0) / n_class2
        
#         covs = []
#         for p in a:
#             x_i = p
#             r = x_i - mu3
#             vec = np.expand_dims(r, axis=0)
#             cov_i = vec * vec.T
#             covs.append(cov_i)
#         covs = np.array(covs)
        
#         p_class_3_data = np.expand_dims(p_class3_given_data, axis=1)
#         p_class_3_data = np.expand_dims(p_class_3_data, axis=1)
#         cov3 = np.sum(p_class_3_data * covs, axis=0) / n_class3
#     # for scores later on
#     p_data_given_class1 = stats.multivariate_normal.pdf(a, mean=mu1, cov=cov1)
#     p_data_given_class2 = stats.multivariate_normal.pdf(a, mean=mu2, cov=cov2)
#     p_data_given_class3 = stats.multivariate_normal.pdf(a, mean=mu3, cov=cov3)

#     p_class1_given_data = p_data_given_class1*p_class1
#     p_class2_given_data = p_data_given_class2*p_class2
#     p_class3_given_data = p_data_given_class3*p_class3

#     summ = p_class1_given_data + p_class2_given_data + p_class3_given_data
#     p_class1_given_data = p_class1_given_data / summ
#     p_class2_given_data = p_class2_given_data / summ
#     p_class3_given_data = p_class3_given_data / summ
        
#     return mu1, mu2, mu3, cov1, cov2, cov3, p_class1_given_data, p_class2_given_data, p_class3_given_data