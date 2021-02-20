import logging

import numpy as np

from sklearn.model_selection import train_test_split, KFold

from domain.model_controller import ModelController


class DataTransformer:

    def __init__(self, config):
        self.models = []
        for model_config in config["first_layer"]:
            self.models.append(ModelController(clf=model_config["clf"], params=model_config["params"],
                                               is_eval=model_config["eval"], eval_metric=model_config["eval_metric"]))
        self.models_data = dict()

    def transform(self, config, train_data, test_data):
        logging.info("All first layer models will be initialized.")
        target_column = train_data[config["target_column"]]
        del train_data[config["target_column"]]
        X_train, X_valid, y_train, y_valid = train_test_split(train_data, target_column,
                                                              test_size=config["validation_size"],
                                                              random_state=config["random_state"])

        ntrain = X_train.shape[0]
        nvalid = X_valid.shape[0]
        kfold = KFold(n_splits=config["kfold"], random_state=config["random_state"], shuffle=True)

        train = None
        valid = None
        test = None

        for i, model in enumerate(self.models):
            logging.info(type(model.clf).__name__ + " started.")
            cur_train, cur_valid, cur_test = get_transform(model, X_train.values, y_train.values, X_valid.values,
                                                           y_valid.values,
                                                           test_data.values, ntrain, nvalid, config["num_classes"],
                                                           kfold,
                                                           eval_set=[(X_valid, y_valid)])
            self.models_data[type(model.clf).__name__] = (cur_train, cur_valid, cur_test)
            if i == 0:
                train = cur_train
                valid = cur_valid
                test = cur_test
            else:
                train = np.concatenate((train, cur_train), axis=1)
                valid = np.concatenate((valid, cur_valid), axis=1)
                test = np.concatenate((test, cur_test), axis=1)

        return train, y_train, valid, y_valid, test, self.models_data


def get_transform(clf, x_train, y_train, x_valid, y_valid, x_test, ntrain, nvalid, num_classes, kfold, eval_set=None):

    train = np.zeros((ntrain, num_classes,))
    valid = np.zeros((nvalid, num_classes,))

    for i, (train_index, test_index) in enumerate(kfold.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        train[test_index] = clf.predict_proba(x_te)
    if clf.eval:
        clf.train_eval(x_train, y_train, eval_set, clf.eval_metric)
    else:
        clf.train(x_train, y_train)
    valid[:] = clf.predict_proba(x_valid)
    test = clf.predict_proba(x_test)

    return train.reshape(ntrain, -1), valid.reshape(nvalid, -1), test
