# NBA Stats Clustering
# Copyright Matthew Strong, 2019

# code to get player data from a season, any season
import numpy as np
import pandas as pd

from nba_api.stats.endpoints import leaguedashplayerstats

from nba_api.stats.static import players


def by_season(year,debug=False, games_thresh=0, headers=None):
    '''
    given a year, gets all players in a dataframe

    args:

    `year`: str: year, like `2019-20`

    `debug`: bool: if we want to print out dataframe

    `games_thresh`: int: take out players who player <= `games_thresh` games
    '''
    # given the year, gets all of the players in 1 call
    
    p = leaguedashplayerstats.LeagueDashPlayerStats(per_mode_detailed='PerGame',
                                                    season='2019-20', headers=headers)
    data = p.league_dash_player_stats.data['data']
    # all the columns we want for clustering
    columns = np.array(['PlayerID', 'GP','PTS','AST','REB','STL','BLK','TOV',
                'FT_PCT', 'FG_PCT', 'FG3_PCT', 'FTA', 'FGA', 'FG3A', 'MIN', 'PLUS_MINUS'])
    r = len(data)
    c = len(columns)
    # store players into matrix with r players (rows) and c columns
    players_arr = np.zeros((r,c))

    # non-empty data
    if data != []:
        # these are the indices that are used for the columns, this is what stats.nba.com decided
        num = np.array([0,5,29,22,21,24,25,23,18,12,15,17,11,14,9,30])
        for i,player in enumerate(data):
            try:
                # print if player's name
                name = players.find_player_by_id(player[0])['full_name']
                print(f'{name} found.')
            except:
                # name not found
                print('Player name not found')
            for j,n in enumerate(num):
                # populate matrix with data
                players_arr[i,j] = player[n]
    # convert matrix and cols to df to store in csv on disk
    df = pd.DataFrame(players_arr, columns=columns)
    to_int = ['PlayerID', 'GP']
    # only columns that need to be int
    df[to_int] = df[to_int].astype(int)
    df_filtered = df[df['GP'] > games_thresh] 
    if debug:
        print(df_filtered)
    # also export to csv, so you only have to run it once.
    df_filtered.to_csv(f'data/{year}_nba_players.csv')

    return df_filtered
