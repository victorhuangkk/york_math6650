import os
from src.pipeline_script import dataProcessor
from src.ml_model import classifier
import pandas as pd

"""
this the executor
"""


def data_preparation():
    model = dataProcessor()
    training_X, test_X, trainint_Y, test_Y = model.data_loading()
    numerical_features, categorical_features, \
    binary_features, unvaried_features = model.feature_selection()
    scaler = model.numerical_processing()

    x_train_num = pd.DataFrame(scaler.transform(training_X[numerical_features]),
                               columns=numerical_features)
    x_test_num = pd.DataFrame(scaler.transform(test_X[numerical_features]),
                              columns=numerical_features)

    for cat in categorical_features:
        temp_df = model.categorical_process(training_X, cat)
        x_train_num = pd.concat([x_train_num, temp_df], axis=1)

    for cat in categorical_features:
        temp_df = model.categorical_process(test_X, cat)
        x_test_num = pd.concat([x_test_num, temp_df], axis=1)
    x_test_num_fill = model.control_test_match(x_train_num, x_test_num)

    x_train_num[binary_features] = training_X[binary_features].astype('int32')
    x_test_num_fill[binary_features] = test_X[binary_features].astype('int32')
    return x_train_num, trainint_Y, x_test_num_fill, test_Y


def main():
    os.chdir('/Users/16477/Desktop/Data Science/final_project')
    x_train_num, trainint_Y, x_test_num_fill, test_Y = data_preparation()

    model = classifier(training_x=x_train_num, training_y=trainint_Y,
                       test_x=x_test_num_fill, test_y=test_Y)
    model.xgboost()


if __name__ == '__main__':
    main()
