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

- This repository now allows you to specify whatever year, and get all of the active players from that year.
