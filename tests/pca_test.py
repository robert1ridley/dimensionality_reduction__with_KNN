import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models import Principal_component_analysis, Knn


def calculate_accuracy(predicted_y, actual_y):
  correct_prediction_count = 0
  incorrect_prediction_count = 0
  for i in range(len(predicted_y)):
    if predicted_y[i] == actual_y[i]:
      correct_prediction_count += 1
    else:
      incorrect_prediction_count += 1
  accuracy = correct_prediction_count / (correct_prediction_count + incorrect_prediction_count)
  print(accuracy)


def main(argv):
  print(argv)
  train_file = argv[0]
  test_file = argv[1]
  k_dimension_value = int(argv[2])
  train_PCA = Principal_component_analysis()
  train_PCA.get_data(train_file)
  train_PCA.get_x_and_y()
  train_PCA.calc_co_variance()
  train_PCA.calc_weight_matrix(k_dimension_value)
  train_PCA.calc_new_feature_space()
  train_set_reduced_features = train_PCA.reduced_feature_space
  train_set_y = train_PCA.y

  test_PCA = Principal_component_analysis()
  test_PCA.get_data(test_file)
  test_PCA.get_x_and_y()
  test_PCA.calc_co_variance()
  test_PCA.calc_weight_matrix(k_dimension_value)
  test_PCA.calc_new_feature_space()
  test_set_reduced_features = test_PCA.reduced_feature_space
  test_set_y = test_PCA.y
  knn = Knn()
  predictions = []
  for index in range(len(test_set_reduced_features)):
    prediction = knn.make_prediction(train_set_reduced_features, test_set_reduced_features[index], train_set_y)
    predictions.append(prediction)
  calculate_accuracy(predictions, test_set_y)


if __name__ == '__main__':
    main(sys.argv[1:])