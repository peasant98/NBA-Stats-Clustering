# main cluster file
import numpy as np
# reads the csv into dataframe before clustering
import CreateClusterData
import DimReduce

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

from nba_api.stats.static import players

class NBACluster():
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        # reduce dimensions if was requested

    def init_data_from_df(self, year, dim_vals, normalize=False, dim_reduce=False, dim_num=3):
        # normalize is set to false, but you should do it 
        self.dim_vals = dim_vals
        self.names, self.x, self.df, self.ordered_dims = CreateClusterData.create_cluster_data(year, dim_vals, normalize)
        # initializes the data from the dataframe
        print(self.dim_vals)
        if dim_reduce:
            self.x, self.df, self.ordered_dims = DimReduce.pca(self.x, dim_num)
            self.dim_vals = self.ordered_dims
        print(f'{self.names.shape[0]} unique points, each with dimension {self.x.shape[-1]}')

    def fit(self, eps):
        # fit the data
        pass
    def test(self, test_data):
        # test the data
        pass
    def get_labels(self):
        # return labels from engine.
        return self.labels
    def plot(self, disp_names=False, thresh=0.8):
        self.color_labels = [f'Group {i+1}' for i in range(self.num_clusters)]
        groups = [[] for i in range(self.num_clusters)]
        group_labels = [[] for i in range(self.num_clusters)]
        for i,p in enumerate(self.x):
            groups[self.labels[i]].append(p)
            group_labels[self.labels[i]].append(self.labels[i])
        groups = np.array(groups)
        if len(self.dim_vals) == 1:
            pass
        elif len(self.dim_vals) == 2:
            fig, ax = plt.subplots()
            # for i,group in enumerate(groups):
                # g = np.array(group)
                # plt.scatter(g[::,0], g[::,1], c=self.labels, label=self.color_labels)
            ax.scatter(self.x[::,0], self.x[::,1], c=self.labels)
            # plt.xlabel('f')
            ax.set_xlabel(self.ordered_dims[0])
            ax.set_ylabel(self.ordered_dims[1])
            if disp_names:
                for i,p in enumerate(self.x):
                    if p[0] > thresh or p[1] > thresh:
                        name_obj = players.find_player_by_id(self.names[i])
                        if name_obj != None:
                            name = name_obj['full_name']
                            ax.text(p[0],p[1], name)
        elif len(self.dim_vals) == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(xs=self.x[::,0], ys=self.x[::,1], zs=self.x[::,2], c=self.labels)
            ax.set_xlabel(self.ordered_dims[0])
            ax.set_ylabel(self.ordered_dims[1])
            ax.set_zlabel(self.ordered_dims[2])
            if disp_names:
                for i,p in enumerate(self.x):
                    if p[0] > thresh or p[1] > thresh or p[2] > thresh:
                        name_obj = players.find_player_by_id(self.names[i])
                        if name_obj != None:
                            name = name_obj['full_name']
                            ax.text(p[0],p[1],p[2], name)
        plt.show()
        # works for 1,2, and 3d data
        
        # plot the result of everything