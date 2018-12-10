import numpy as np

class Knn(object):

  def __init__(self):
    self.distances = []
    self.targets = None

  def make_prediction(self, X_train, x_test, y_train):
    self.distances = []
    for i in range(len(X_train)):
      # first we compute the euclidean distance
      distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
      # add it to list of distances
      self.distances.append((distance, i))

      # sort the list
    distances = sorted(self.distances)

    # make a list of the k neighbors' targets
    index = distances[0][1]
    self.target = y_train[index]

    # return most common target
    return self.target

