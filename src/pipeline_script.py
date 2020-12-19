import pandas as pd
import numpy as np
from sklearn import preprocessing


class dataProcessor:
    """
    module to process data and generate model ready dataset
    """
    def __init__(self):
        self.__train__ = pd.read_csv("aps_failure_training_set.csv")
        self.__test__ = pd.read_csv("aps_failure_test_set.csv")

    def data_loading(self):
        na_train_df = self.__train__.replace('na', np.nan, regex=True)
        na_test_df = self.__test__.replace('na', np.nan, regex=True)

        na_train_df['num_class'] = np.where(na_train_df['class'] == "pos", 1, -1)
        na_test_df['num_class'] = np.where(na_test_df['class'] == "pos", 1, -1)

        del na_train_df['class']
        del na_test_df['class']

        na_train_df_na_fill = na_train_df.fillna(na_train_df.mode().iloc[0])
        na_test_df_na_fill = na_test_df.fillna(na_test_df.mode().iloc[0])

        na_train_df_na_fill = na_train_df_na_fill.astype("float64")
        na_test_df_na_fill = na_test_df_na_fill.astype("float64")

        y_train = na_train_df_na_fill['num_class']
        y_test = na_test_df_na_fill['num_class']

        x_train = na_train_df_na_fill.loc[:, na_train_df_na_fill.columns != 'num_class']
        x_test = na_test_df_na_fill.loc[:, na_test_df_na_fill.columns != 'num_class']
        self.__x_train__ = x_train

        return x_train, x_test, y_train, y_test

    def feature_selection(self):
        numerical_features = []
        categorical_features = []
        binary_features = []
        unvaried_features = []
        for colname in self.__x_train__.columns.values:
            if len(np.unique(self.__x_train__[colname].to_list())) == 2:
                binary_features.append(colname)
            elif len(np.unique(self.__x_train__[colname].to_list())) < 2:
                unvaried_features.append(colname)
            elif 2 < len(np.unique(self.__x_train__[colname].to_list())) <= 47:
                categorical_features.append(colname)
            else:
                numerical_features.append(colname)

        self.__numerical_features__ = numerical_features

        return numerical_features, categorical_features, binary_features, unvaried_features

    def numerical_processing(self):
        x_train_num_features = self.__x_train__[self.__numerical_features__]
        scaler = preprocessing.StandardScaler().fit(x_train_num_features)
        return scaler

    @staticmethod
    def categorical_process(data_frame, cat_name):
        data_frame[cat_name] = data_frame[cat_name].astype('category')
        dummy_df = pd.get_dummies(data_frame[cat_name])
        dummy_df.columns = [*map(lambda t: str(cat_name) + str(t), list(dummy_df.columns.values))]
        return dummy_df

    @staticmethod
    def control_test_match(x_train_num, x_test_num):
        missing_cols = set(x_train_num.columns) - set(x_test_num.columns)
        # Add a missing column in test set with default value equal to 0
        for c in missing_cols:
            x_test_num[c] = 0
        # Ensure the order of column in the test set is in the same order than in train set
        x_test_num = x_test_num[x_train_num.columns]
        return x_test_num

    @staticmethod
    def add_binary_feature(DF, binary_feature):
        return 1

