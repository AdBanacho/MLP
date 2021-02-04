import numpy as np
import matplotlib.pylab as plt

from abc import abstractmethod

from mlp.data import Dataset
from mlp.metrics import Accuracy, Cost


class Optimizer:
    def __init__(self):
        pass

    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def backpropagation(self, *args):
        pass


class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate, batch_size,
                 epoch, loss, activation, l2, size_of_valid):
        super().__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.loss = loss
        self.activation = activation
        self.l2 = l2
        self.size_of_valid = size_of_valid

    def train(self, model, training_data):
        dataset = Dataset()
        train_accuracy = Accuracy()
        valid_accuracy = Accuracy()
        train_cost = Cost(self.loss, self.l2, model)
        valid_cost = Cost(self.loss, self.l2, model)

        for j in range(self.epoch):
            np.random.shuffle(training_data.T)
            size = int(training_data.shape[1] * (1-self.size_of_valid))

            train_input = training_data[1:, :size]
            train_output = training_data[0:1, :size]
            train_input = dataset.normalization(train_input, 255)
            train_output = dataset.hot_one(train_output, 10)

            valid_input = training_data[1:, size:]
            valid_output = training_data[0:1, size:]
            valid_input = dataset.normalization(valid_input, 255)
            valid_output = dataset.hot_one(valid_output, 10)

            for k in range(0, len(train_output.T), self.batch_size):
                mini_input = train_input[:, self.batch_size: 2*self.batch_size]
                mini_output = train_output[:, self.batch_size: 2*self.batch_size]
                self.update(model, mini_input, mini_output, len(train_output.T))

            train_model_predict = model.forward(train_input).T
            valid_model_predict = model.forward(valid_input).T

            train_accuracy.evaluate(train_output.T, train_model_predict)
            valid_accuracy.evaluate(valid_output.T, valid_model_predict)
            train_cost.evaluate(train_output.T, train_model_predict)
            valid_cost.evaluate(train_output.T, valid_model_predict)

            print("Epoch: ", j + 1)
            print("Accuracy train: ", train_accuracy.history[j],
                  "Accuracy valid: ", valid_accuracy.history[j],)
            print("Cost train: ", train_cost.history[j],
                  "Cost valid: ", valid_cost.history[j])

        epochs = range(1, self.epoch + 1)
        plt.plot(epochs, train_accuracy.history, 'g', label='Training accuracy')
        plt.plot(epochs, valid_accuracy.history, 'b', label='Validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        epochs = range(1, self.epoch+1)
        plt.plot(epochs, train_cost.history, 'g', label='Training loss')
        plt.plot(epochs, valid_cost.history, 'b', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def update(self, model, mini_input, mini_output, n):
        delta_w = []
        delta_b = []
        for p in range(self.batch_size):
            de_w, de_b = self.backpropagation(
                model, mini_input[:, p:p + 1], mini_output[:, p:p + 1])
            for i in range(len(model.layers)):
                delta_b.append(np.zeros(de_b[i].shape))
                delta_w.append(np.zeros(de_w[i].shape))
                delta_w[i] += de_w[i]
                delta_b[i] += de_b[i]

        for m in range(1, len(model.layers)):
            model.layers[m].weights = \
                (1 - self.learning_rate * self.l2 / n) \
                * model.layers[m].weights \
                - (self.learning_rate / len(mini_output.T)) \
                * delta_w[m]

            model.layers[m].bias = \
                model.layers[m].bias \
                - (self.learning_rate / len(mini_output.T)) \
                * delta_b[m]

    def backpropagation(self, model, input_x, output):

        zs = []
        activations = []
        delta_b = []
        delta_w = []

        for num_of_layers in range(0, len(model.layers)):
            input_x = model.layers[num_of_layers].forward(input_x)
            zs.append(input_x)
            input_x = model.activations[num_of_layers].activation(input_x)
            activations.append(input_x)
            delta_b.append(np.zeros((input_x.shape[0], 1)))
            delta_w.append(np.zeros((input_x.shape[0], input_x.shape[1])))

        delta = self.loss.delta(activations[-1], output, zs[-1])

        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].T)

        for m in range(2, len(model.layers)):
            delta = np.dot(model.layers[-m+1].weights.T, delta) \
                    * self.activation.activation_derivative(zs[-m])
            delta_b[-m] = delta
            delta_w[-m] = np.dot(delta, activations[-m-1].T)

        return delta_w, delta_b
