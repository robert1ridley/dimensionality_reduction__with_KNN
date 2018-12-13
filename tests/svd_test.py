import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import get_raw_data, preprocess_data
from models import Knn, Svd
from pandas import DataFrame as df


def get_data(filepath):
  raw_data = get_raw_data(filepath)
  preprocessed_data = preprocess_data(raw_data)
  row_length = len(preprocessed_data[0])
  data = df(preprocessed_data)
  return data, row_length


def get_x_and_y(data, row_length):
  X = data.loc[:,0:(row_length-2)].values
  y = data.loc[:,row_length-1].values
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


def main(argv):
  train_file = argv[0]
  test_file = argv[1]
  k_dimension_value = int(argv[2])
  data, row_length = get_data(train_file)
  X, y = get_x_and_y(data, row_length)
  train_svd = Svd(X, k_dimension_value)
  train_params = train_svd.calc_svd()
  test_data, row_length = get_data(test_file)
  test_X, test_y = get_x_and_y(test_data, row_length)
  test_svd = Svd(test_X, k_dimension_value)
  test_params = test_svd.calc_svd()
  knn = Knn()
  predictions = []
  for index in range(len(test_params)):
    prediction = knn.make_prediction(train_params, test_params[index], y)
    predictions.append(prediction)
  calculate_accuracy(predictions, test_y)


if __name__ == '__main__':
    main(sys.argv[1:])
