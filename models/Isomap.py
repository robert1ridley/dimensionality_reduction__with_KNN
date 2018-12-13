import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn import metrics
from scipy.sparse.csgraph import floyd_warshall as fw


class Isomap(object):

  def __init__(self, in_feature_size, in_neighbors):
    self.lower_dim_feature_space_size = int(in_feature_size)
    self.number_of_neighbors = int(in_neighbors)
    self.floyd_shortest_path = None


  def calculate_distance_matrix(self, x, y):
    d = metrics.pairwise_distances(x, y)
    return d


  def cal_B(self, D):
    (n1, n2) = D.shape
    DD = np.square(D)
    Di = np.sum(DD, axis=1) / n1
    Dj = np.sum(DD, axis=0) / n1
    Dij = np.sum(DD) / (n1 ** 2)
    B = np.zeros((n1, n1))
    for i in range(n1):
      for j in range(n2):
        B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)
    return B


  def get_isomap(self, D):
    K = self.lower_dim_feature_space_size
    self.floyd_shortest_path = fw(D)
    B = self.cal_B(self.floyd_shortest_path)
    Be, Bv = np.linalg.eigh(B)
    Be_sort = np.argsort(-Be)
    Be = Be[Be_sort]
    Bv = Bv[:, Be_sort]
    Bez = np.diag(Be[0:K])
    Bvz = Bv[:, 0:K]
    Z = np.dot(np.sqrt(Bez), Bvz.T).T
    return Z
