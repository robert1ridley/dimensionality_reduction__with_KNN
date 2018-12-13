import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pandas import DataFrame as df
from models import Isomap, Knn
from utils import get_raw_data, preprocess_data


def get_data(filepath):
  raw_data = get_raw_data(filepath)
  preprocessed_data = preprocess_data(raw_data)
  row_length = len(preprocessed_data[0])
  data = df(preprocessed_data)
  return data, row_length


def get_x_and_y(data, row_length):
  X = data.ix[:,0:(row_length-2)].values
  y = data.ix[:,row_length-1].values
  return X, y


def calculate_accuracy(predicted_y, actual_y):
  correct_prediction_count = 0
  incorrect_prediction_count = 0
  for i in range(len(predicted_y)):
    if predicted_y[i] == actual_y[i]:
      correct_prediction_count += 1
    else:
      incorrect_prediction_count += 1
  accuracy = correct_prediction_count / (correct_prediction_count + incorrect_prediction_count)
  print("ACCURACY: " + str(accuracy))


def main(args):
  train_file = args[0]
  test_file = args[1]
  K_val = args[2]
  number_of_neighbors = args[3]
  data, row_length = get_data(train_file)
  X, y = get_x_and_y(data, row_length)
  iso_train = Isomap(K_val, number_of_neighbors)
  train_dist = iso_train.calculate_distance_matrix(X, X)
  train_iso = iso_train.get_isomap(train_dist)

  test_data, test_row_l = get_data(test_file)
  test_X, test_y = get_x_and_y(test_data, test_row_l)

  knn = Knn()
  predictions = []
  test_iso = Isomap(K_val, number_of_neighbors)
  for i in range(len(test_X)):
    this_item_x = np.vstack((X, test_X[i]))
    item_dist = test_iso.calculate_distance_matrix(this_item_x, this_item_x)
    item_iso = test_iso.get_isomap(item_dist)
    prediction = knn.make_prediction(train_iso, item_iso, y)
    predictions.append(prediction)
  acc = calculate_accuracy(predictions, test_y)
  print(acc)


if __name__ == '__main__':
  args = sys.argv[1:]
  main(args)
