from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np

from mlp.activations import Activation, LinearActivation, Sigmoid, Softmax
from mlp.initializers import GaussianInitializer, UniformInitializer, \
    OneInitializer
from mlp.optimizers import StochasticGradientDescent
from mlp.losses import MeanSquaredError, CrossEntropy
from mlp.data import Dataset
from mlp.metrics import Accuracy, Cost, Precision, Recall, F1, TabOfPredict


class Layer(ABC):

    @abstractmethod
    def forward(self, input_x):
        pass


class InputLayer(Layer):
    def __init__(self, input_features):
        self.size = input_features
        super().__init__()

    def forward(self, input_x):
        return input_x


class Dense(Layer):
    def __init__(self, input_features, output_features,
                 weight_initializer, bias_initializer):
        super().__init__()
        self.weights = weight_initializer.initialize(
            (output_features, input_features))
        self.bias = bias_initializer.initialize(
            (output_features, 1))

    def forward(self, input_x):
        return self.weights @ input_x + self.bias


class Mlp:
    def __init__(self):
        self.layers: List[Layer] = []
        self.activations: List[Activation] = []

    def add_module(self, module: Union[Layer, Activation]):
        if isinstance(module, Layer):
            self.layers.append(module)
        else:
            self.activations.append(module)

    def forward(self, input_x):
        for i in range(len(self.layers)):
            input_x = self.layers[i].forward(input_x)
            input_x = self.activations[i].activation(input_x)
        return input_x


if __name__ == '__main__':

    dataset = Dataset()
    initializer = GaussianInitializer(0, 1)
    loss_SGD = MeanSquaredError()
    activation_SGD: Sigmoid = Sigmoid()

    train_data, test_input, test_output = \
        dataset.load_data(size_of_test=0.2)

    l2 = 0.0
    SGD = StochasticGradientDescent(
        learning_rate=3.0, batch_size=20,
        epoch=50, loss=loss_SGD, activation=activation_SGD, l2=l2, size_of_valid=0.2)

    model = Mlp()
    model.add_module(InputLayer(784))
    model.add_module(LinearActivation())
    model.add_module(Dense(784, 30, initializer, initializer))
    model.add_module(Sigmoid())
    model.add_module(Dense(30, 10, initializer, initializer))
    model.add_module(Sigmoid())

    SGD.train(model, train_data.T)

    predict_data = (model.forward(test_input.T)).T

    accuracy = Accuracy()
    cost = Cost(loss_SGD, l2, model)
    precision = Precision()
    recall = Recall()
    f1 = F1()
    tab_of_predict = TabOfPredict()

    accuracy.evaluate(test_output, predict_data)
    cost.evaluate(test_output, predict_data)
    recall.evaluate(test_output, predict_data)
    precision.evaluate(test_output, predict_data)
    f1.evaluate(recall.history[0], precision.history[0])
    tab_of_predict.evaluate(test_output, predict_data)
    print("Accuracy: ", accuracy.history[0], "\n",
          "Cost: ", cost.history[0], "\n",
          "Recall: ", "\n", recall.history[0], "\n",
          "Precision: ", "\n", precision.history[0], "\n",
          "F1 score: ", "\n", f1.history[0], "\n",
          "Table of prediction: ", "\n", tab_of_predict.history[0])


