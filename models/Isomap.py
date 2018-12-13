import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn import metrics
from scipy.sparse.csgraph import floyd_warshall as fw
from operator import itemgetter


class Isomap(object):

  def __init__(self, in_feature_size, in_neighbors):
    self.lower_dim_feature_space_size = int(in_feature_size)
    self.number_of_neighbors = int(in_neighbors)


  def calculate_distance_matrix(self, x, y):
    return metrics.pairwise_distances(x, y)


  def get_isomap(self, D):
    K = self.lower_dim_feature_space_size
    floyd_shortest_path = fw(D)
    eig_val, eig_vec = np.linalg.eig(floyd_shortest_path)
    eigen_pairs = []
    for i in range(len(eig_val)):
      eigen_pair = (np.abs(eig_val[i]), eig_vec[:,i])
      eigen_pairs.append(eigen_pair)
    eigen_pairs.sort(key=itemgetter(0))
    eigen_pairs.reverse()
    intermediate_list = []
    for k in range(K):
      intermediate_list.append(eigen_pairs[k][1].reshape(len(D-1), 1))
    Z = np.hstack(intermediate_list)
    return Z
