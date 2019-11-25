# NBA Stats Clustering
# Copyright Matthew Strong, 2019

import numpy as np
import pandas as pd
import time

from nba_api.stats.endpoints import playerdashboardbyyearoveryear
from nba_api.stats.static import players
from multiprocessing import Process, Value, Array

N_VAL = 12

# get data for specific season from every player out of active players today who was active.
def get_id_from_players_list(entry):
    return entry['id']

def place_player(ind, player_id, year, arr, m, n):
    player_year = playerdashboardbyyearoveryear.PlayerDashboardByYearOverYear(player_id=player_id,
                                            season=year)
    with arr.get_lock():
        print(f'{player_id} found.')
        np_arr = np.frombuffer(arr.get_obj()) # mp_arr and arr share the same memory
        # make it two-dimensional
        b = np_arr.reshape((m,n))
        if player_year.overall_player_dashboard.data['data'] != []:
            player_season = player_year.overall_player_dashboard.data['data'][0]
            # season is available
            # games played
            b[ind, 0] = player_id
            num = np.array([5,29,22,21,24,25,23,18,12,15,17])
            # gp, pts, ast, reb, stl, blk, tov, ft_pct, fg_pct, fg3_pct, fta
            for i,val in enumerate(num):
                b[ind, i+1] = player_season[val]

def get_season_data(year, debug=True):
    n = N_VAL
    all_players = players.get_active_players()
    # ids of all active players
    ids = np.array(list(map(get_id_from_players_list, all_players)))
    m = len(ids)
    jobs = []
    v = int(m / 15) + 1
    split_ids = np.array_split(ids, v)
    arr = Array('d', m*n)
    np_arr = np.frombuffer(arr.get_obj())
    ind = 0
    for split in split_ids:
        jobs = []
        for player_id in split:
            # get important stats from each year
            p = Process(target=place_player, args=(ind, player_id, year, arr, m, n))
            jobs.append(p)
            p.start()
            ind+=1
        for j in jobs:
            j.join()
    b = np_arr.reshape((m,n))
    # from m x n matrix b, construct dataframe
    # m rows - each person
    columns = np.array(['PlayerID','GP','PTS','AST','REB','STL','BLK','TOV',
                'FT_PCT', 'FG_PCT', 'FG3_PCT', 'FTA'])
    # gp, pts, ast, reb, stl, blk, tov, ft_pct, fg_pct, fg3_pct, fta
    df = pd.DataFrame(b, columns=columns)
    to_int = ['PlayerID', 'GP', 'PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV']
    df[to_int] = df[to_int].astype(int)
    if debug:
        print(df)
    # also export to csv
    df.to_csv(f'{year}_nba_players.csv')
    return df
