from abc import abstractmethod

import numpy as np


class Metric:
    def __init__(self):
        self.history = []

    @abstractmethod
    def evaluate(self, target, prediction):
        pass

    @property
    def last(self):
        return self.history[-1]


class Accuracy(Metric):
    def evaluate(self, target, prediction):
        accuracy = 0.0
        for i in range(len(prediction)):
            if np.argmax(target[i]) == np.argmax(prediction[i]):
                accuracy += 1
        self.history.append(np.round(100 * accuracy / len(target),3))


class Cost(Metric):
    def __init__(self, loss, l2, model):
        super().__init__()
        self.loss = loss
        self.l2 = l2
        self.model = model

    def evaluate(self, target, prediction):
        cost = 0.0
        weights = 0.0
        for i in range(len(prediction)):
            cost += self.loss.loss(prediction[i], target[i])
        for m in range(1, len(self.model.layers)):
            weights += np.sum(self.model.layers[m].weights ** 2)
        cost += 0.5 * self.l2 * weights
        self.history.append(np.round(cost / len(prediction), 3))


class TabOfPredict(Metric):
    def evaluate(self, target, prediction):
        tab_of_predict = np.zeros((len(target.T), len(target.T)))
        for i in range(len(target)):
            tab_of_predict[np.argmax(prediction[i])][np.argmax(target[i])] += 1
        self.history.append(tab_of_predict)


class Precision(Metric):
    def evaluate(self, target, prediction):
        table_of_predict = np.zeros((len(target.T), len(target.T)))
        for i in range(len(target)):
            table_of_predict[np.argmax(prediction[i])][np.argmax(target[i])] += 1
        precision = np.sum(table_of_predict, axis=1)

        for i in range(len(target.T)):
            precision[i] = np.round(table_of_predict[i][i] / precision[i], 3)

        self.history.append(precision)


class Recall(Metric):
    def evaluate(self, target, prediction):
        table_of_predict = np.zeros((len(target.T), len(target.T)))
        for i in range(len(target)):
            table_of_predict[np.argmax(prediction[i])][np.argmax(target[i])] += 1
        recall = np.sum(table_of_predict, axis=0)

        for i in range(len(target.T)):
            recall[i] = np.round(table_of_predict[i][i] / recall[i], 3)

        self.history.append(recall)


class F1(Metric):
    def evaluate(self, target, prediction):
        f1_score = np.zeros(len(target))
        for i in range(target.shape[0]):
            f1_score[i] = np.round(2 * target[i] * prediction[i] / (prediction[i] + target[i]), 3)

        self.history.append(f1_score)
