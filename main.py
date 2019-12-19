# NBA Stats Clustering
# Copyright Matthew Strong, 2019

# good ol' numpy
import numpy as np
import argparse

from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import playerdashboardbyyearoveryear
from nba_api.stats.static import players
from nba_api.stats.library.parameters import Season

import data.get_player_overall as get_players

# importing my custom clustering methods.
import clustering.GMM as nba_gmm
import clustering.Hierarchical as nba_hierarchical
import clustering.KMeans as nba_kmeans

import matplotlib.pyplot as plt

# 3 total different clustering methods that I initially used
# pts,ast,reb  reb,stl,blk  fta,fga,fg3a
if __name__ == '__main__':
    # 
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=str, default='none', help='season year')
    parser.add_argument('--num_clusters', type=int, default=3, help='num clusters max')
    
    opt = parser.parse_args()
    # get clustering and dimension methods from cmd line
    # the season is selected, make api call to get players' stats from that seasons
    if opt.season != 'none':
        df = get_players.by_season(opt.season)

    # if the user decides to do less than 3 clusters for whatever reason
    num_clusters = max(opt.num_clusters,3)
    # default year
    year = '2019-20'
    interactive_plot = False
    # default columns, select from strings:
    #'PlayerID', 'GP','PTS','AST','REB','STL','BLK','TOV', 'FT_PCT', 
    # 'FG_PCT', 'FG3_PCT', 'FTA', 'FGA', 'FG3A', 'MIN', 'PLUS_MINUS'
    # select any combination of them
    cols = ['PTS', 'AST', 'REB']
    # if we want to normalize the data for each method
    normalize = True
    # if names of players should be plotted that are on the higher end of the results
    plot_names = False
    # arrays for later displaying the error
    km_elbows = []
    km_simple_random_elbows = []
    km_simple_extreme_elbows = []
    gmm_elbows = []
    hierarchical_elbows = []
    k_nums = []
    # IMPORTANT 

    # all clustering methods inherit from base class NBACluster, which 
    # provides methods such as plotting, showing assignments, getting data,
    # which are universal regardless of clustering method.

    # go for k=3 to 9 by default, unless num_clusters from cmd line is different
    for k in range(3, num_clusters+1):
        ## k means sklearn with kmeans++
        k_nums.append(k)
        nba = nba_kmeans.NBAKMeans(k)
        # get data from dataframe give year and columns, also good to normalize
        nba.init_data_from_df(year, cols, normalize=normalize)
        # fit with kmeans++, 300 max iterations 0.001 tolerance till we stop
        nba.fit('k-means++', 300, 0.0001)
        km_elbows.append(nba.ssd)
        # this function diplays every players' group as well as that centroid's mean
        nba.text_display_cluster()

        # plot the player. Can plot single name as well
        # if disp_names is True, display players with high values on one or many of the axes
        nba.plot(single_name='LeBron James', disp_names=plot_names, interactive=interactive_plot)
        # ## k means simple
        
        nba1 = nba_kmeans.NBAKMeansSimple(k)
        # a
        nba1.init_data_from_df(year, cols, normalize=normalize)

        # true for random initialization or points for cluster centers.
        nba1.fit(True, 0.0001)
        km_simple_random_elbows.append(nba1.ssd)
        nba1.plot(single_name='LeBron James', disp_names=plot_names, interactive=interactive_plot)

        nba2 = nba_kmeans.NBAKMeansSimple(k)
        nba2.init_data_from_df(year, cols, normalize=normalize)

        # # # true for random initialization.
        # fits given points farthest away from mean
        nba2.fit(False, 0.0001)
        km_simple_extreme_elbows.append(nba2.ssd)

        # plot funcion!
        nba2.plot(single_name='LeBron James', disp_names=plot_names, interactive=interactive_plot)
        
        # # ## gaussian mixture model

        
        nba3 = nba_gmm.NBAGMM(k)
        # specify k
        nba3.init_data_from_df(year, cols, normalize=normalize)


        nba3.fit()
        # fit given 100 iterations
        gmm_elbows.append(nba3.ssd)
        # # # print(nba.get_labels())
        nba3.plot(single_name='LeBron James', disp_names=plot_names, interactive=interactive_plot)
        

        # ## hierarchical
        
        nba4 = nba_hierarchical.NBAHierarchical(k)
        # uses ward linkage, which uses euclidean distance to calculate variances by default.
        nba4.init_data_from_df(year, cols, normalize=normalize)
        # fit with euc. and ward linkage
        nba4.fit('euclidean')
        hierarchical_elbows.append(nba4.ssd)
        nba4.plot(single_name='LeBron James', disp_names=plot_names, interactive=interactive_plot)
    # plot k vs sum of squared distances from the center 
    all_elbows = [km_elbows, km_simple_random_elbows, km_simple_extreme_elbows, gmm_elbows, hierarchical_elbows]
    elbow_strings = ['KM', 'KM-Simple-Random', 'KM-Simple-Extreme', 'GMM', 'Hierarchical']
    # plot for each method, for all k's selected, in a certain year, for a certain group of fields.
    for i in range(len(all_elbows)):
        plt.plot(k_nums, all_elbows[i])
        plt.xlabel('K')
        plt.ylabel(f'Sum of Squared Distance')
    plt.legend(elbow_strings)
    plt.title(f'SSD for Clustering-{year}-{cols}')
    plt.savefig(f'img/SSD-{year}-{cols}')


    
