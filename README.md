# NBA Stats Clustering

A repo with various tools and tricks for clustering of NBA Players. 

Includes:

- Gaussian Mixture Models

- K-Means Clustering

- Hierarchical Agglomerative Clustering

- PCA

- Yearly Stats

## Requires

1. `pandas`
2. `numpy`
3. `nba-api`
4. `argparse`
5. `scipy`
6. `mpl_toolkits`
7. `matplotlib`
8. `sklearn`


Also, needs Python3.6 or above due to `f`-strings.

## Usage

- `git clone https://github.com/peasant98/NBA-Stats-Clustering/`

- `cd NBA-Stats-Clustering`
- `python3.6 main.py`, to run the file and make a bunch of plots, which will be stored in the `img/` folder. However, if you want to interact with the cool 3d plots, have line 40 in `main.py` be `interactive_plot = True` instead of `interactive_plot = False`.

- The above command will run clustering on points, rebounds, and assists for this season from k=3 to 9 on
k-means (3 different initialization methods), gmm, and hierarchical clustering

- `python3.6 main.py --season=<year>`, where `year` needs to be of the form `20XX-XX+1`; example: `2019-20`.

- This repository now allows you to specify whatever year, and get all of the active players from that year.

- Additionally, the `img/` folder shows a bunch of pictures of clustering results among different fields. clustering methods, and values of `k`.