# NBA Stats Clustering
# Copyright Matthew Strong, 2019

import numpy as np
import argparse

from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import playerdashboardbyyearoveryear
from nba_api.stats.static import players
from nba_api.stats.library.parameters import Season

import data.get_player_overall as get_players

if __name__ == '__main__':
    # 
    parser = argparse.ArgumentParser()
    parser.add_argument('--clustering_option', type=str, default='kmeans', help='clustering method')
    parser.add_argument('--dim_reduce_option', type=str, default=None, help='dimension reduction method')
    parser.add_argument('--season', type=str, default='2019-20', help='season year')
    
    parser.add_argument('--val_list', nargs='+', help='<Required> Set flag')
    opt = parser.parse_args()
    # get clustering and dimension methods from cmd line
    clustering_method = opt.clustering_option
    dim_reduce_method = opt.dim_reduce_option
    # for working with active players:
    all_players = players.get_active_players()
    # # given year, get all players
    df = get_players.get_season_data(opt.season)
    # # also store into csv 
    # clustering method
        

    
