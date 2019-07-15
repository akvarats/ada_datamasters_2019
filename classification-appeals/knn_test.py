import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances


def get_distance_matrix():

    X = [[1], [2], [3], [4], [5], [6], [7], [8]]

    distance = lambda x, y: x + y if x != y else 0

    return pairwise_distances(X, metric=distance)


if __name__ == "__main__":

    print(get_distance_matrix())

    X = get_distance_matrix()

    y = [0, 1, 1, 0, 1, 2, 1, 0]

    neigh = KNeighborsClassifier(n_neighbors=2, metric="precomputed")

    neigh.fit(X, y)

    print(neigh.predict_proba([[1, 2, 2, 1, 1, 3, 3, 1]]))






