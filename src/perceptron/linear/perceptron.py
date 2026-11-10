import random

from perceptron.activations import sigmoid, der_sigmoid
from perceptron.losses import mse, der_mse


class Perceptron:

    def __init__(self, n_input):
        if n_input <= 0:
            raise ValueError("Inputs num must be greater than zero")

        self.bias = random.random()

        self.weights = [
            random.random()
            for _ in range(n_input)
        ]

    def forward(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("Input dimension mismatch")

        z = self.bias

        for inp, weight in zip(inputs, self.weights):
            z += inp * weight

        return sigmoid(z)

    def train(self, inputs, target, learning_rate):
        prediction = self.forward(inputs)

        gradient = (
            der_mse(prediction, target)
            * der_sigmoid(prediction)
        )

        for i in range(len(self.weights)):
            self.weights[i] -= (
                learning_rate * gradient * inputs[i]
            )

        self.bias -= learning_rate * gradient

        return mse(prediction, target)