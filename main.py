# NBA Stats Clustering
# Copyright Matthew Strong, 2019

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

if __name__ == '__main__':
    # 
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=str, default='none', help='season year')
    
    opt = parser.parse_args()
    # get clustering and dimension methods from cmd line
    if opt.season != 'none':
        df = get_players.by_season(opt.season)
        # simply perform clustering
    # here are the clustering methods
    k = 5
    year = '2019-20'
    cols = ['PTS', 'AST', 'REB']
    normalize = True
    ## k means

    
    nba = nba_kmeans.NBAKMeans(k)
    nba.init_data_from_df(year, cols, normalize=normalize)

    nba.fit('k-means++', 300, 0.0001)

    nba.plot(True)
    
    

    ## k means simple
    
    nba1 = nba_kmeans.NBAKMeansSimple(k)
    nba1.init_data_from_df(year, cols, normalize=normalize)

    # true for random initialization.
    nba1.fit(True, 0.0001)

    nba1.plot(True)

    nba2 = nba_kmeans.NBAKMeansSimple(k)
    nba2.init_data_from_df(year, cols, normalize=normalize)

    # true for random initialization.
    nba2.fit(False, 0.0001)

    nba2.plot(True)
    
    

    ## gaussian mixture model

    
    nba3 = nba_gmm.NBAGMM(k)
    nba3.init_data_from_df(year, cols, normalize=normalize)


    nba3.fit()

    # print(nba.get_labels())


    nba3.plot(disp_names=True)
    

    ## hierarchical
    
    nba4 = nba_hierarchical.NBAHierarchical(k)
    nba4.init_data_from_df(year, cols, normalize=normalize)
    nba4.fit('euclidean')
    nba4.plot(True)
    

        

    
