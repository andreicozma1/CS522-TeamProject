import random
import numpy as np
from time import time
from tqdm import tqdm


def sigmoid(z):
    """
    Implementing the sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of sigmoid function
    """
    return sigmoid(z) * (1 - sigmoid(z))


class BPNN:

    def __init__(self, sizes, random_init=True, verbose=False):
        """
        This class implements a Feedforward Neural Network using the
        stochastic gradient descent learning algorithm, and backpropagation
        is used to calculate the gradients.
        `sizes` contains the number of neurons in the
        respective layers of the network based on the index
        (ex, index 0 is input layer, last index is output layer)
        The biases and weights for the network are initialized randomly
        """
        self.verbose = verbose
        self.num_layers = len(sizes)
        self.sizes = sizes
        # Initialize weights and biases using random Gaussian distribution
        # Starting from the first hidden layer initialize weights in an array of shape [len_prev_layer, len_curr_layer]
        if random_init:
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = [
                np.random.uniform(-1, 1, size=(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]

        # Initialize biases except for the input layer
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

    def feedforward(self, a):
        """
        Return the output of the network if ``a`` is input.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train(self, training_data, max_epochs, batch_size, learning_rate, evaluation_data, evaluation_treshold=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        """
        if self.verbose:
            print(f"# Training MLP Network: Sizes={self.sizes}\t"
                  f"Epochs={max_epochs}\t"
                  f"Batch-Size={batch_size}\t"
                  f"Learning-Rate={learning_rate}")

        training_data = list(training_data)
        evaluation_data = list(evaluation_data)

        n = len(training_data)
        n_test = len(evaluation_data)

        if self.verbose:
            print(f' - Training Data Len: {n}')
            print(f' - Validation Data Len: {n_test}')
            print("# Epochs:")

        evaluation_scores = [0]
        evaluation_deltas = []
        t0 = time()
        for j in tqdm(range(max_epochs)):
            # Shuffle the training data
            random.shuffle(training_data)

            # Construct a set of mini batches to update for the training
            batches = [
                training_data[k:k + batch_size]
                for k in range(0, n, batch_size)]

            # For each mini batch run the update function
            for batch in batches:
                self.update_batch(batch, learning_rate)

            # Print mini-batch evalutaion on testing data

            evaluation_correct = self.evaluate(evaluation_data)
            evaluation_score = evaluation_correct / n_test

            delta_evaluation_score = (
                evaluation_score - evaluation_scores[-1]) if len(evaluation_scores) != 0 else 0
            evaluation_deltas.append(delta_evaluation_score)

            evaluation_deltas_avg = np.average(evaluation_deltas)

            if self.verbose:
                print(f"\t{j + 1}. Correct {evaluation_correct}/{n_test}\t"
                      f"(score: {round(evaluation_score, 4)}\t"
                      f"delta: {round(delta_evaluation_score, 4)}\t"
                      f"delta_avg: {round(evaluation_deltas_avg, 4)})")

            evaluation_scores.append(evaluation_score)

            if evaluation_treshold is not None and evaluation_deltas_avg < evaluation_treshold:
                break

        t1 = time()
        conv_time = t1 - t0
        evaluation_scores = evaluation_scores[1:]
        evaluation_deltas[0] = 0
        if self.verbose:
            print(
                f"Converged in {len(evaluation_scores)} epochs with accuracy {evaluation_scores[-1]} took {round(conv_time, 2)} sec")

        if evaluation_treshold is None:
            evaluation_deltas = None

        return evaluation_scores[-1], evaluation_scores, evaluation_deltas, conv_time

    def update_batch(self, batch, learning_rate):
        """
        Apply gradient descent to a batch using backpropagation to
        update the network's weights and biases.
        """
        # Construct a numpy array of weights and biases
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        n_batch = len(batch)

        for x, y in batch:
            # Use back propagation to get the change in weights and biases
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Update the weights and biases
        self.weights = [w - (learning_rate / n_batch) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / n_batch) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        The tuple returned represents the gradient for the cost function C_x.
        """
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(
            activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # for each layer of neurons
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives
        partial C_x / partial a
        for the output activations.
        """
        return (output_activations - y)
