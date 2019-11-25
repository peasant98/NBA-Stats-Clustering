# NBA Stats Clustering

A repo with various tools and tricks for clustering of NBA Players. Currently is implemented for active NBA players only.

Right now, you can grab some data from the regular season

## Requires

1. `pandas`
2. `numpy`
3. `nba-api`

## Usage

- `git clone https://github.com/peasant98/NBA-Stats-Clustering/`

- `cd NBA-Stats-Clustering`

- `python3.6 main.py --season=<year>`, where `year` needs to be of the form `20XX-XX+1`; example: `2019-20`.

- This repository only fetches active players (this year), so if you select a previous year, you will only get the subset of players who are still active, and played in that previous year. For example, 2003-04 will barely have any players (LeBron and some others). 


