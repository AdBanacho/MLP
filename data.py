import csv

import numpy as np


class Dataset:
    def load_data(self, size_of_test):
        data_path = \
            "C:/Users/Adrian/PycharmProjects/Infosys/mlp/data/"
        data = np.loadtxt(data_path + "train.csv",
                                delimiter=",")

        size = int(data.shape[0] * (1 - size_of_test))
        num_of_classes = 10
        train_data = data[:size]
        test_input = data[size:, 1:]
        test_output = data[size:, 0:1]
        test_input = self.normalization(test_input, 255)
        test_output = (self.hot_one(test_output.T, num_of_classes)).T

        return train_data, test_input, test_output

    def normalization(self, data, scal):
        return data / scal

    def hot_one(self, y_train, num_of_class):
        y = np.zeros((num_of_class, len(y_train.T)))
        for i in range(len(y_train.T)):
            y[int(y_train[0][i])][i] = 1
        return y


class DatasetIterator:
    def __init__(self, dataset, step):
        self.dataset = dataset
        self.step = step
        self.index = 0
