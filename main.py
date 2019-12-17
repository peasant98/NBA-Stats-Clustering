# NBA Stats Clustering
# Copyright Matthew Strong, 2019


# 1)   Code should be fully commented.  This can be in markdown cells describing the 
# algorithm, or simply in standard comments.
# 2)   Any relevant citations for outside resources used MUST BE PRESENT.  Your code will 
# be run through anti-plagiarism software.
# 3)   Clearly note any algorithms used.  If your code differs from conventions used in class, 
# make note of that in comments.
# 4)   In this submission, you will be evaluated for whether or not your project’s content 
# included a rigorous implementation of CSCI4022 techniques.  If you choose to use 
# default packages instead of variants of the code from in class, you must clearly comment 
# in and describe why you are using that package, and how its implementation compares 
# to in-class variants.

# Some percentage of the project points will go towards clarity of steps taken, 
# so make sure you explain any default choices that the 
# sklearn/statsmodels/whatever packages you're using may take. 


import numpy as np
import argparse

from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import playerdashboardbyyearoveryear
from nba_api.stats.static import players
from nba_api.stats.library.parameters import Season

import data.get_player_overall as get_players

import clustering.GMM as nba_gmm
import clustering.Hierarchical as nba_hierarchical
import clustering.KMeans as nba_kmeans

import matplotlib.pyplot as plt

# 3 total different clustering methods
# pts,ast,reb  reb,stl,blk  fta,fga,fg3a
if __name__ == '__main__':
    # 
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=str, default='none', help='season year')
    parser.add_argument('--num_clusters', type=int, default=3, help='num clusters max')
    
    opt = parser.parse_args()
    # get clustering and dimension methods from cmd line
    if opt.season != 'none':
        df = get_players.by_season(opt.season)
        # simply perform clustering
    # here are the clustering methods
    # if the user decides to do less than 3 clusters for whatever reason
    num_clusters = max(opt.num_clusters,3)
    year = '2018-19'
    cols = ['PTS', 'MIN', 'PLUS_MINUS']
    normalize = True
    plot_names = True
    km_elbows = []
    km_simple_random_elbows = []
    km_simple_extreme_elbows = []
    gmm_elbows = []
    hierarchical_elbows = []
    k_nums = []
    for k in range(3, num_clusters+1):
        ## k means
        k_nums.append(k)
        nba = nba_kmeans.NBAKMeans(k)
        nba.init_data_from_df(year, cols, normalize=normalize)

        nba.fit('k-means++', 300, 0.0001)
        km_elbows.append(nba.ssd)
        nba.plot(plot_names)
        
        # ## k means simple
        
        nba1 = nba_kmeans.NBAKMeansSimple(k)
        nba1.init_data_from_df(year, cols, normalize=normalize)

        # true for random initialization.
        nba1.fit(True, 0.0001)
        km_simple_random_elbows.append(nba1.ssd)
        nba1.plot(plot_names)

        nba2 = nba_kmeans.NBAKMeansSimple(k)
        nba2.init_data_from_df(year, cols, normalize=normalize)

        # # # true for random initialization.
        nba2.fit(False, 0.0001)
        km_simple_extreme_elbows.append(nba2.ssd)


        nba2.plot(plot_names)
        
        # # ## gaussian mixture model

        
        nba3 = nba_gmm.NBAGMM(k)
        nba3.init_data_from_df(year, cols, normalize=normalize)


        nba3.fit()
        gmm_elbows.append(nba3.ssd)
        # # # print(nba.get_labels())
        nba3.plot(plot_names)
        

        # ## hierarchical
        
        nba4 = nba_hierarchical.NBAHierarchical(k)
        nba4.init_data_from_df(year, cols, normalize=normalize)
        nba4.fit('euclidean')
        hierarchical_elbows.append(nba4.ssd)
        nba4.plot(plot_names)
    all_elbows = [km_elbows, km_simple_random_elbows, km_simple_extreme_elbows, gmm_elbows, hierarchical_elbows]
    elbow_strings = ['KM', 'KM-Simple-Random', 'KM-Simple-Extreme', 'GMM', 'Hierarchical']
    for i in range(len(all_elbows)):
        plt.plot(k_nums, all_elbows[i])
        plt.xlabel('K')
        plt.ylabel(f'Sum of Squared Distance')
    plt.legend(elbow_strings)
    plt.title(f'SSD for Clustering-{year}-{cols}')
    plt.savefig(f'img/SSD-{year}-{cols}')


    
