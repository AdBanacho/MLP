import numpy as np

from abc import ABC, abstractmethod


class Initializer(ABC):
    @abstractmethod
    def initialize(self, size):
        pass


class UniformInitializer:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def initialize(self, size):
        return np.random.randint(self.start, self.stop, size)


class GaussianInitializer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def initialize(self, size):
        return np.random.normal(self.mean, self.std, size)


class ZeroInitializer(Initializer):
    def initialize(self, size):
        return np.zeros(size)


class OneInitializer(Initializer):
    def initialize(self, size):
        return np.ones(size)
