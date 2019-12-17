# NBA Stats Clustering
# Copyright Matthew Strong, 2019

import numpy as np
import pandas as pd
import time

from nba_api.stats.endpoints import playerdashboardbyyearoveryear
from nba_api.stats.endpoints import leaguedashplayerstats

from nba_api.stats.static import players
from multiprocessing import Process, Value, Array

N_VAL = 12

def by_season(year,debug=False, games_thresh=0):
    # given the year, gets all of the players in 1 call
    p = leaguedashplayerstats.LeagueDashPlayerStats(per_mode_detailed='PerGame',
                                                    season=year)
    data = p.league_dash_player_stats.data['data']
    columns = np.array(['PlayerID', 'GP','PTS','AST','REB','STL','BLK','TOV',
                'FT_PCT', 'FG_PCT', 'FG3_PCT', 'FTA', 'FGA', 'FG3A', 'MIN', 'PLUS_MINUS'])
    r = len(data)
    c = len(columns)
    players_arr = np.zeros((r,c))
    if data != []:
        # valid season.
        num = np.array([0,5,29,22,21,24,25,23,18,12,15,17,11,14,9,30])
        for i,player in enumerate(data):
            try:
                name = players.find_player_by_id(player[0])['full_name']
                print(f'{name} found.')
            except:
                print('Player name not found')
            for j,n in enumerate(num):
                players_arr[i,j] = player[n]
    df = pd.DataFrame(players_arr, columns=columns)
    to_int = ['PlayerID', 'GP']
    df[to_int] = df[to_int].astype(int)
    df_filtered = df[df['GP'] > games_thresh] 
    if debug:
        print(df_filtered)
    # also export to csv
    df_filtered.to_csv(f'data/{year}_nba_players.csv')
    return df_filtered
# get data for specific season from every player out of active players today who was active.
