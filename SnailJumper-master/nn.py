import numpy as np
import math


class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        self.w = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.normal(0, 1, size=(layer_sizes[i + 1], layer_sizes[i]))   # normal random
            self.w.append(w)
            b = np.zeros((layer_sizes[i + 1], 1))     # zero
            self.b.append(b)

    def activation(self, x, function='sigmoid'):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        if function == 'softmax':
            return np.exp(x) / np.exp(x).sum()
        else:
            return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # print("shape of x ", np.shape(x))
        a = x
        z = self.w[0] @ a + self.b[0]
        a = self.activation(z)
        z = self.w[1] @ a + self.b[1]
        a = self.activation(z, 'softmax')
        return a