# NBA Stats Clustering

A repo with various tools and tricks for clustering of NBA Players. 

Includes:

- Gaussian Mixture Models

- K-Means Clustering

- Hierarchical Agglomerative Clustering

- PCA

- Yearly Stats

## Requires

```sh
pandas
numpy
nba-api
argparse
scipy
mpl_toolkits
matplotlib
sklearn

```

Also, needs Python3.6 or above due to f-strings.

## Usage
```sh
git clone https://github.com/peasant98/NBA-Stats-Clustering/
cd NBA-Stats-Clustering/
python3.6 main.py
```
This will run the file and make a bunch of plots, which will be stored in the `img/` folder.

- The above command will run clustering on points, rebounds, and assists for this season from k=3 to 9 on
k-means (3 different initialization methods), gmm, and hierarchical clustering

For more advanced usage:

```sh
python3.6 main.py --season=<year> --num_clusters=<num> --interactive=<interactive_bool>
```

`year` needs to be of the form `20XX-XX+1`; example: `2019-20`. `num`  from `num_clusters` needs to be an `int`, and specifies the max number of clusters the application will go to (from `3` to `num`). `interactive_bool` needs to be a `bool`, and specifies if you want to see an interactive plot on each clustering for each value of `k` clusters.

This repository now allows you to specify whatever year, and get all of the active players from that year.

Additionally, the `img/` folder shows a bunch of pictures of clustering results among different fields. clustering methods, and values of `k`.

Due to some updates in the NBA stats API, there is the variable `HEADERS` defined in `constants.py` to ensure no connection times out.
