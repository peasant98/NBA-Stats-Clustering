import numpy as np

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import playercareerstats


x = playergamelog.PlayerGameLog(2544)
print(x)
career = playercareerstats.PlayerCareerStats(player_id=2544)
print(career)   