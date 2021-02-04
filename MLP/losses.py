import numpy as np

from abc import ABC, abstractmethod

from mlp.activations import Sigmoid


class Loss(ABC):
    @abstractmethod
    def loss(self, *args):
        pass

    @abstractmethod
    def derivative_loss(self, *args):
        pass

    @abstractmethod
    def delta(self, *args):
        pass


class MeanSquaredError(Loss):
    def loss(self, y_hat, y):
        return 0.5 * np.linalg.norm(y_hat - y) ** 2

    def derivative_loss(self, y_hat, y):
        return y_hat - y

    def delta(self, y_hat, y, z):
        sigmoid = Sigmoid()
        return self.derivative_loss(y_hat, y) \
               * sigmoid.activation_derivative(z)


class CrossEntropy(Loss):
    def loss(self, y_hat, y):
        # ep = 1e-7
        return np.sum(np.nan_to_num(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat)))
        # return -np.sum(y @ np.log(y_hat + ep))

    def derivative_loss(self):
        pass

    def delta(self, y_hat, y, z):
        return y_hat - y
