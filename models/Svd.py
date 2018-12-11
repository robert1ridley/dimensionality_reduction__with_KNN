import numpy as np


class Svd(object):

  def __init__(self, in_A, in_k):
    self.A = in_A
    self.k = in_k


  def calc_svd(self):
    U, S, V = np.linalg.svd(self.A, full_matrices=True)
    k = self.k
    sig_diag = np.zeros(shape=(k, k))
    for i in range(k):
      sig_diag[i, i] = S[i]

    reduced_u = U[:len(U), :k]
    reduced_s_v = np.dot(sig_diag, V[:k, :len(V)])
    return np.dot(reduced_u, reduced_s_v)
