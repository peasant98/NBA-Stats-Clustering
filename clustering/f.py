import numpy as np

def dist(x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    # z, z1 = get_extreme_points(5, dfy, ['views', 'likes'])
    # get extreme points initialization method for kmeans    
def get_extreme_points(k, df, cols):
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
    arr = [(dist(center, val), val) for val in X]
    # most extreme points in terms of distance from x_bar or mean
    res = sorted(arr, key=lambda tup: tup[0])[::-1][:k]
    return np.array(list(np.array(res)[:,1])), X

get_extreme_points()