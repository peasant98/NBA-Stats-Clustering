# NBA Stats Clustering
# Copyright Matthew Strong, 2019

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import playerdashboardbyyearoveryear
from nba_api.stats.static import players
from multiprocessing import Process, Value, Array

# get data for specific season from every player out of active players today who was active.
def get_id_from_players_list(entry):
    return entry['id']

def place_player(ind, player_id, year, arr, m, n):
    with arr.get_lock():
        np_arr = np.frombuffer(arr.get_obj()) # mp_arr and arr share the same memory
        # make it two-dimensional
        b = np_arr.reshape((m,n))
        b[ind, 0] = player_id

    # player_year = playerdashboardbyyearoveryear.PlayerDashboardByYearOverYear(player_id=id,
                                            # season=year)

def get_season_data(year):
    n = 5
    all_players = players.get_active_players()
    # ids of all active players
    ids = np.array(list(map(get_id_from_players_list, all_players)))
    m = len(ids)
    df = pd.DataFrame({'id': [],
                        'name': []})
    jobs = []
    print(m*n)
    arr = Array('d', m*n)
    np_arr = np.frombuffer(arr.get_obj())
   
    for ind, player_id in enumerate(ids):
        # get important stats from each year
        p = Process(target=place_player, args=(ind, player_id, year, arr, m, n))
        jobs.append(p)
        p.start()
    for j in jobs:
        j.join()
    b = np_arr.reshape((m,n))
    print(b)
    print('done!')


get_season_data('2019-20')