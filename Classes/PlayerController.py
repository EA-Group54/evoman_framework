from evoman.controller import Controller
import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class PlayerController(Controller):
    def __init__(self, nodes):
        # Number of hidden nodes
        self.nodes = nodes

    def set(self, params, inputs):
        # Getting bias and weights for hidden layer
        self.bias1 = params[:self.nodes]
        weightslice = self.nodes * inputs + self.nodes
        self.weight1 = params[self.nodes:weightslice].reshape(self.nodes, inputs)

        # Bias and weights for output layer
        self.bias2 = params[weightslice:weightslice + 5]
        self.weight2 = params[weightslice + 5:].reshape(5, self.nodes)


    def control(self, params, cont):
        # Computing output in hidden layer
        output = sigmoid(self.weight1 @ params + self.bias1)
        output = sigmoid(self.weight2 @ output + self.bias2).flatten()

        actions = []
        # Looping through outputs
        for action in output:
            actions.append(int(action >= 0.8))

        return actions
