import random
import numpy as np


def set_by_matrix(m):
    return set([tuple(x) for x in m])

def has_converged(prev_c, curr_c):
    return set_by_matrix(prev_c) == set_by_matrix(curr_c)


def cluster_points(X, centres):
    clusters = {}
    for x in X:
        bestmukey = min([(i, np.linalg.norm(x[1] - centres[i]))
                         for i, _ in enumerate(centres)], key=lambda x: x[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters


def reevaluate_centers(clusters):
    centres = []
    keys = sorted(clusters.keys())
    for k in keys:
        normalized = [x[1] for x in clusters[k]]
        centres.append(np.mean(normalized, axis=0))
    return centres

class kmeans(object):
    """docstring for kmeans"""

    def __init__(self, num_clusters, corpus):
        self.num = num_clusters
        self.corpus = corpus



    def fit(self, X):
        prev_c = [x[1] for x in random.sample(X, self.num)]
        curr_c = [x[1] for x in random.sample(X, self.num)]
        while not has_converged(prev_c, curr_c):
            prev_c = curr_c
            clusters = cluster_points(X, curr_c)
            curr_c = reevaluate_centers(clusters)
        
        self.clusters = clusters
    

    def get_clusters(self):
        pretty_data = []
        print(len(self.clusters))
        for name, cluster in self.clusters.items():
            data  = []
            for item in cluster:
                data.append(self.corpus[item[0]])
            pretty_data.append(data)
        
        return pretty_data


    def predict(self, x, y):
        pass











