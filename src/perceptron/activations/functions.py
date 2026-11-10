import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def der_sigmoid(output):
    return output * (1 - output)