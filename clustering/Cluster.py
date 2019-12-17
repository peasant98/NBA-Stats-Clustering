# main cluster file
import numpy as np
# reads the csv into dataframe before clustering
import clustering.CreateClusterData
import clustering.DimReduce

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nba_api.stats.static import players

class NBACluster():
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        # reduce dimensions if was requested

    def init_data_from_df(self, year, dim_vals, normalize=True):
        # normalize can be set to false, but you should not do it 
        self.year = year
        self.dim_vals = dim_vals
        self.cols = dim_vals
        self.reduced = False
        self.names, self.x, self.df, self.ordered_dims = clustering.CreateClusterData.create_cluster_data(year, dim_vals, normalize)
        # initializes the data from the dataframe
        # pca dimension reduction if there are 4 or more dimensions on which we want to cluster
        if len(self.dim_vals) > 3:
            self.x, self.df, self.ordered_dims = clustering.DimReduce.pca(self.x, 3)
            self.dim_vals = self.ordered_dims
            self.reduced = True
        print(f'{self.names.shape[0]} unique points, each with dimension {self.x.shape[-1]}')

    def fit(self, eps):
        '''
        fit function.
        '''
        # fit the data
        pass
    def get_labels(self):
        '''
        gets the labels from the fitting of the classification.
        '''
        # return labels from engine.
        return self.labels
    def text_display_cluster(self):
        '''
        displays all of the groups that every player is in after `fit()` is run, as well
        their corresponding centroid.
        '''
        for i,p in enumerate(self.x):
            name_obj = players.find_player_by_id(self.names[i])
            if name_obj != None:
                name = name_obj['full_name']
                print(f'{name}: Group {self.labels[i]} with centroid {self.centroids[self.labels[i]]}')

    def plot(self, disp_names=False, thresh=0.8, single_name='', interactive=False):
        '''
        plots the cluster points.

        `disp_names`: `bool`: selects whether to display some players' names or not.

        `thresh`: `float`, between `0` and `1`: given each dimensions max value, take `thresh * 100%` of that to show names.

        `single_name`: `str`: If the user wants to see where a specific player is classified, they can do so here.

        `interactive`: `bool`: If the user wants to be able to interact with the plot after each clustering `fit` is run.
        
        '''
        self.priority_name_index = -1
        player = players.find_players_by_full_name(single_name)
        if len(player) == 1:
            # there is a valid player with the name
            self.p_id = player[0]['id']
            index = np.where(np.array(self.names)==self.p_id)
            if len(index) == 1:
                # player does in fact exist
                self.priority_name_index = index[0][0]

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
            dim1_thresh = np.max(self.x[::,0]) * thresh
            dim2_thresh = np.max(self.x[::,1]) * thresh

            if disp_names:
                for i,p in enumerate(self.x):
                    if p[0] > dim1_thresh or p[1] > dim2_thresh:
                        name_obj = players.find_player_by_id(self.names[i])
                        if name_obj != None:
                            name = name_obj['full_name']
                            ax.text(p[0],p[1], name)
            for i,p in enumerate(self.x):
                if i==self.priority_name_index:
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
            dim1_thresh = np.max(self.x[::,0]) * thresh
            dim2_thresh = np.max(self.x[::,1]) * thresh
            dim3_thresh = np.max(self.x[::,2]) * thresh

            if disp_names:
                for i,p in enumerate(self.x):
                    if p[0] > dim1_thresh or p[1] > dim2_thresh or p[2] > dim3_thresh:
                        name_obj = players.find_player_by_id(self.names[i])
                        if name_obj != None:
                            name = name_obj['full_name']
                            ax.text(p[0],p[1],p[2], name)

            for i,p in enumerate(self.x):
                if i==self.priority_name_index:
                    name_obj = players.find_player_by_id(self.names[i])
                    if name_obj != None:
                        name = name_obj['full_name']
                        ax.text(p[0],p[1],p[2], name)
        is_dr = '' if not self.reduced else '-with-PCA'
        rounded_ssd = np.round(self.ssd, 4)
        title = f'{self.method}-k={self.num_clusters}-{self.cols}-{self.year}{is_dr}'
        plt.title(f'{title}-ssd={rounded_ssd}')
        plt.savefig(f'img/{title}')
        if interactive:
            plt.show()
        plt.close()