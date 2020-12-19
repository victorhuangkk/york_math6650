from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import imblearn
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree


class classifier:
    def __init__(self, training_x, training_y,
                 test_x, test_y, oversampling=True,
                 method='logistic regression'):
        self.__training_x__ = training_x
        self.__traing_y__ = training_y
        self.__test_x__ = test_x
        self.__test_y__ = test_y
        self.__oversampling = oversampling
        self.__method__ = method
        self.__X_train_smote__ = 0
        self.__y_train_smote__ = 0

    def over_sampling(self):
        oversample = imblearn.over_sampling.SMOTE(random_state=42)
        os_data_X, os_data_y = oversample.fit_resample(self.__training_x__.to_numpy(),
                                                       self.__traing_y__.to_numpy())

        return pd.DataFrame(data=os_data_X, columns=self.__training_x__.columns), \
               pd.DataFrame(data=os_data_y, columns=['num_class'])

    def logistic_regression(self):
        x_train_smote, y_train_smote = self.over_sampling()

        clf = LogisticRegression(max_iter=1000)

        clf.fit(x_train_smote,
                   y_train_smote.values.ravel())

        self.reporting(clf, "Logistic Regression")

    def support_vector(self):
        x_train_smote, y_train_smote = self.over_sampling()
        clf = sklearn.svm.SVC(kernel='rbf', probability=True)
        clf.fit(x_train_smote, y_train_smote.values.ravel())

        # Predict the response for test dataset
        self.reporting(clf, "SVM")

    def classfication_tree(self):
        x_train_smote, y_train_smote = self.over_sampling()
        clf = sklearn.tree.DecisionTreeClassifier(random_state=42)
        clf.fit(x_train_smote,
                y_train_smote.values.ravel())

        self.reporting(clf, "Classification Tree")

        plt.figure()
        plot_tree(clf, filled=True, max_depth=3)
        plt.show()

    def randomForest(self):
        # clf = RandomForestClassifier(random_state=0)
        x_train_smote, y_train_smote = self.over_sampling()

        calibrated_forest = CalibratedClassifierCV(
            base_estimator=RandomForestClassifier(n_estimators=10))
        param_grid = {
            'base_estimator__max_depth': [2, 4, 6, 8]}
        search = GridSearchCV(calibrated_forest, param_grid, cv=5)
        search.fit(x_train_smote,
                   y_train_smote.values.ravel())
        self.reporting(search, "Random Forest")

    def xgboost(self):
        model = XGBClassifier()
        x_train_smote, y_train_smote = self.over_sampling()
        model.fit(x_train_smote,
                  y_train_smote.values.ravel())

        # self.reporting(model, "XGBoost")

    def stackModel(self):
        x_train_smote, y_train_smote = self.over_sampling()
        estimators = [
            ('rf', RandomForestClassifier(random_state=42)),
            ('xgboost', XGBClassifier())
        ]
        clf = StackingClassifier(
            estimators=estimators, final_estimator=LogisticRegression()
        )
        clf.fit(x_train_smote, y_train_smote.values.ravel())
        # self.reporting(clf, "StackingClassifier")

    def reporting(self, clf, model_name):
        accu = clf.score(self.__test_x__, self.__test_y__)
        print("Accuracy of {} is {}".format(model_name, accu, ))
        y_pred = clf.predict_proba(self.__test_x__)

        y_scores = y_pred[:, -1:]
        auc = roc_auc_score(self.__test_y__, y_scores)
        print("AUC of {} is {}".format(model_name, auc))
        predict_y = clf.predict(self.__test_x__)
        print(confusion_matrix(self.__test_y__, predict_y))
        print(sklearn.metrics.classification_report(self.__test_y__, predict_y))

        fpr, tpr, thresholds = roc_curve(self.__test_y__.to_numpy(), y_scores)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of {}'.format(model_name))
        plt.show()
