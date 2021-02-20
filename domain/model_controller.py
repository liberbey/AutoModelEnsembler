from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


class ModelController(object):
    def __init__(self, clf, params=None, is_eval=False, eval_metric=None):
        self.clf = eval(clf)(**params)
        self.eval = is_eval
        self.eval_metric = eval_metric

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def train_eval(self, x_train, y_train, eval_set, eval_metric, early_stopping_rounds=10):
        return self.clf.fit(x_train, y_train, early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric,
                            eval_set=eval_set)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

    def feature_importances(self, x, y):
        print(self.clf.fit(x, y).feature_importances_)