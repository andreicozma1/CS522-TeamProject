import numpy as np


def get_activation_func(name: str):
    if name == 'relu':
        return lambda x: np.maximum(x, 0)
    elif name == 'sigmoid':
        return lambda x: 1 / (1 + np.exp(-x))
    elif name == 'tanh':
        return lambda x: (2 / (1 + np.exp(-2 * x))) - 1
    elif name == 'leaky_relu':
        return lambda x: 0.01 * x if x < 0 else x
    elif name == 'softmax':
        return lambda x: softmax(x)
    raise Exception("Activation Function Not Implemented")


def softmax(x):
    # TODO
    pass
