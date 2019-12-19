# NBA Stats Clustering
# Copyright Matthew Strong, 2019

import numpy as np
import pandas as pd

# ,PlayerID,GP,PTS,AST,REB,STL,BLK,TOV,FT_PCT,FG_PCT,FG3_PCT,FTA
def create_cluster_data(year, dim_vals, normalize):
    # get the cluster data from the csv, given n dimensions such as TOV or REB
    # GP is irrelevant
    stats = ['PlayerID', 'GP','PTS','AST','REB','STL','BLK','TOV',
                'FT_PCT', 'FG_PCT', 'FG3_PCT', 'FTA', 'FGA', 'FG3A', 'MIN', 'PLUS_MINUS']
    selection = np.array(list(set(stats).intersection(set(dim_vals))))
    # get intersection of stats and dimension values that the user wants
    df = pd.read_csv(f'data/{year}_nba_players.csv')
    # read csv given the year
    # get selection rows
    player_ids = df['PlayerID'].values
    # numpy matrix of ids
    df = df[selection]
    # get df with the selected dimensions
    # optional min max normalization, True by default
    if normalize:
        df = df[selection].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    arr = df[selection].values
    # values, numpy matrix
    return player_ids, arr, df, selection
