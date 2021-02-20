import json
import logging

import pandas as pd
import numpy as np

from core.helpers import csv_helper
from domain.data_transformer import DataTransformer
from domain.feature_constructor import FeatureConstructor
from domain.model_controller import ModelController
from sklearn.metrics import log_loss


class EnsembleClassifierService:

    @staticmethod
    def run(input_json_file):
        logging.info("Classifier service started.")
        with open(input_json_file, "r", encoding="utf-8") as settings_file:
            config = json.load(settings_file)
        train_data = csv_helper.read_csv_pd(config["train_data"])
        test_data = csv_helper.read_csv_pd(config["test_data"])
        data_transformer = DataTransformer(config)
        transformed_train, y_train, transformed_valid, y_valid, transformed_test, models_data_dict = \
            data_transformer.transform(config, train_data, test_data)

        if config["construct_pca"]:
            logging.info("PCA features will be created.")
            transformed_train, transformed_valid, transformed_test = FeatureConstructor.construct_pca_features(
                config["pca_config"], transformed_train, transformed_valid,
                transformed_test, models_data_dict)
        if config["construct_statistical_features"]:
            logging.info("Statistical features will be created.")
            transformed_train, transformed_valid, transformed_test = FeatureConstructor.construct_statistical_features(
                config, transformed_train, transformed_valid, transformed_test)

        if config["output_firstlayer_results"]:
            logging.info("Writing first layer results to csv files.")
            csv_helper.output_csv_np(transformed_train, "transformed_train.csv")
            csv_helper.output_csv_np(transformed_valid, "transformed_valid.csv")
            csv_helper.output_csv_np(transformed_test, "transformed_test.csv")

        meta_model = ModelController(config["meta_model"]["clf"], params=config["meta_model"]["params"],
                                     is_eval=config["meta_model"]["eval"],
                                     eval_metric=config["meta_model"]["eval_metric"])

        logging.info("Meta model will be initialized.")
        if meta_model.eval:
            meta_model.train_eval(transformed_train, y_train, eval_set=[(transformed_valid, y_valid)],
                                  eval_metric=meta_model.eval_metric)
        else:
            meta_model.train(transformed_train, y_train)
        predictions = meta_model.predict_proba(transformed_test)
        logging.info("Predictions created, results will be written to csv file.")
        csv_helper.output_predictions(config["output_file_name"], predictions)
