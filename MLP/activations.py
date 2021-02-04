import numpy as np

from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def activation(self, *args):
        pass

    @abstractmethod
    def activation_derivative(self, *args):
        pass


class Sigmoid(Activation):
    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def activation_derivative(self, z):
        sigmoid = Sigmoid()
        return sigmoid.activation(z) * (1 - sigmoid.activation(z))


class Relu(Activation):
    def activation(self, z):
        return np.maximum(0, z)

    def activation_derivative(self, z):
        pass


class Tanh(Activation):
    def activation(self, z):
        return (np.exp(z) - np.exp(-z)) \
               / (np.exp(z) + np.exp(-z))

    def activation_derivative(self, z):
        return 4 / ((np.exp(z) + np.exp(-z))**2)


class LinearActivation(Activation):
    def activation(self, x):
        return x

    def activation_derivative(self):
        pass


class Softmax(Activation):
    def activation(self, z):
        e_x = np.exp(z.T - np.max(z, axis=-1))
        return (e_x / e_x.sum(axis=0)).T

    def activation_derivative(self):
        pass
