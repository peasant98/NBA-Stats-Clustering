import numpy as np
import pandas as pd

# ,PlayerID,GP,PTS,AST,REB,STL,BLK,TOV,FT_PCT,FG_PCT,FG3_PCT,FTA
def create_cluster_data(year, dim_vals, normalize):
    # get the cluster data from the csv, given n dimensions such as TOV or REB
    # GP is irrelevant
    stats = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV',
                'FT_PCT', 'FG_PCT', 'FG3_PCT', 'FTA']
    selection = np.array(list(set(stats).intersection(set(dim_vals))))
    df = pd.read_csv(f'data/{year}_nba_players.csv')
    # get selection rows
    player_ids = df['PlayerID'].values
    df = df[selection]
    # just end up normalizing the selection rows 
    if normalize:
        df = df[selection].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    arr = df[selection].values
    return player_ids, arr, df, selection
