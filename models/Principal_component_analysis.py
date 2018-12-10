from utils import get_raw_data, sort_data
from pandas import DataFrame as df
import numpy as np

class Principal_component_analysis(object):

  def __init__(self):
    self.data = None
    self.X = None
    self.y = None
    self.co_variance = None
    self.eigen_pairs = None
    self.weight_matrix = None
    self.reduced_feature_space = None


  def get_data(self, filepath):
    raw_data = get_raw_data(filepath)
    sorted_data = sort_data(raw_data)
    self.data = df(sorted_data)


  def get_x_and_y(self):
    self.X = self.data.ix[:,0:59].values
    self.y = self.data.ix[:,60].values


  def calc_co_variance(self):
    mean_vector = np.mean(self.X, axis=0)
    self.co_variance = (self.X - mean_vector).T.dot((self.X - mean_vector)) / (self.X.shape[0]-1)
    eigen_values, eigen_vectors = np.linalg.eig(self.co_variance)
    eigen_pairs = []
    for i in range (len(eigen_values)):
      eigen_pair = (np.abs(eigen_values[i]), eigen_vectors[:,i])
      eigen_pairs.append(eigen_pair)
    eigen_pairs.sort()
    eigen_pairs.reverse()
    self.eigen_pairs = eigen_pairs


  def calc_weight_matrix(self, comp_number):
    intermediate_list = []
    for i in range(comp_number):
      intermediate_list.append(self.eigen_pairs[i][1].reshape(60, 1))
    self.weight_matrix = np.hstack(intermediate_list)


  def calc_new_feature_space(self):
    self.reduced_feature_space = self.X.dot(self.weight_matrix)
