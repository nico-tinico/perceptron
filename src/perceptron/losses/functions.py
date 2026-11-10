def mse(prediction, target):
    return (prediction - target) ** 2


def der_mse(prediction, target):
    return 2 * (prediction - target)
