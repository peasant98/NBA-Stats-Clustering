import numpy as np

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import playercareerstats
from nba_api.stats.endpoints import leaguedashplayerstats


# x = playergamelog.PlayerGameLog(2544)
# print(x)
# career = playercareerstats.PlayerCareerStats(player_id=2544)
# print(career)   
p = leaguedashplayerstats.LeagueDashPlayerStats(per_mode_detailed='PerGame')
print(p.league_dash_player_stats.data['data'][0])
print(p.league_dash_player_stats.data['headers'])
