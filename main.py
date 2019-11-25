# NBA Stats Clustering
# Copyright Matthew Strong, 2019

import numpy as np
import argparse

from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import playerdashboardbyyearoveryear
from nba_api.stats.static import players
from nba_api.stats.library.parameters import Season



if __name__ == '__main__':
    # 
    parser = argparse.ArgumentParser()
    parser.add_argument('--clustering_option', type=str, default='kmeans', help='clustering method')
    parser.add_argument('--dim_reduce_option', type=str, default=None, help='dimension reduction method')
    parser.add_argument('--season', type=str, default='2003-04', help='season year')
    opt = parser.parse_args()
    # get clustering and dimension methods from cmd line
    clustering_method = opt.clustering_option
    dim_reduce_method = opt.dim_reduce_option
    # for working with active players:
    all_players = players.get_active_players()
    player_year = playerdashboardbyyearoveryear.PlayerDashboardByYearOverYear(player_id=2544,
                                            season='2003-04')
    # possibly useful function:
    # players.find_player_by_id(<id>)
    print(player_year.by_year_player_dashboard.data['headers'])
    print(player_year.by_year_player_dashboard.data['data'])
        
    # player_info = commonplayerinfo.CommonPlayerInfo(player_id=2544)
    # get the ids of the legends

    
