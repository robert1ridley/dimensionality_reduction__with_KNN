import numpy as np
import operator

class Knn(object):

  def __init__(self):
    self.distances = []

  def make_prediction(self, X_train, x_test, y_train):
    self.distances = []
    self.targets = []
    for i in range(len(X_train)):
      distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
      self.distances.append((distance, i))

    distances = sorted(self.distances)

    for i in range(1):
      index = distances[i][1]
      self.targets.append(y_train[index])

    t_counts = {}
    for t in self.targets:
      if t not in t_counts.keys():
        t_counts[t] = 1
      else:
        t_counts[t] +=1
    return max(t_counts.items(), key=operator.itemgetter(1))[0]

