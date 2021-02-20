import pandas as pd
import numpy as np


def read_csv_pd(input_file):
    return pd.read_csv(input_file)


def read_csv_np(input_file):
    return np.loadtxt(input_file, "r", delimiter=",")


def output_csv_np(data, file_name):
    np.savetxt(file_name, data, delimiter=",")


def output_predictions(file_name, predictions):
    with open(file_name, 'w') as f:
        classes = [f'c-{i}' for i in range(len(predictions[0]))]
        f.write(",".join(classes))
        f.write("\n")
        for element in predictions:
            f.write(",".join(str(v) for v in list(element)))
            f.write("\n")