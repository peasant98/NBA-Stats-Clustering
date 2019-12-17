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

    ## k means

    '''
    nba = nba_kmeans.NBAKMeans(5)
    nba.init_data_from_df('2019-20', ['PTS', 'AST', 'REB'], normalize=True)

    nba.fit('k-means++', 300, 0.0001)

    nba.plot(True)
    '''
    

    ## k means simple
    
    nba = nba_kmeans.NBAKMeansSimple(5)
    nba.init_data_from_df('2019-20', ['PTS', 'AST', 'BLK', 'REB'], normalize=True)

    nba.fit(True, 0.0001)

    nba.plot(True)
    
    

    ## gaussian mixture model

    '''
    nba = nba_gmm.NBAGMM(5)
    nba.init_data_from_df('2019-20', ['PTS', 'AST', 'BLK', 'STL'], normalize=True)

    nba.fit()

    # print(nba.get_labels())


    nba.plot(disp_names=True)
    '''

    ## hierarchical
    '''
    nba = nba_hierarchical.NBAHierarchical(5)
    nba.init_data_from_df('2019-20', ['PTS', 'AST', 'BLK', 'STL'], normalize=True)
    nba.fit('euclidean')
    nba.plot(True)
    '''

        

    
