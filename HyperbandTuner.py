from Dataset import *
import cupy as cp
from bpnn_cuda import BPNN
import time
from tqdm import tqdm


class HyperbandTuner:

    def __init__(self, model, num_trials=1, max_epochs=30, eta=3):
        self.model = model
        self.num_trials = num_trials
        self.max_epochs = max_epochs
        self.eta = eta

    def generate_hypers(self, init_nc, init_b, init_lr):
        # Generate all combinations of hyperparameters
        nc = [init_nc]

        for i in range(1, self.eta):
            # don't change the first and last elements
            nc.append(
                [init_nc[0], int(init_nc[1] - init_nc[1] * (i / self.eta)), init_nc[-1]])
            nc.append(
                [init_nc[0], int(init_nc[1] + init_nc[1] * (i / self.eta)), init_nc[-1]])

        print("# Network Configurations")
        for i in nc:
            print(f"\t{i}")

        b = [init_b]
        for i in range(1, self.eta):
            b.append(int(init_b - init_b * (i / self.eta)))
            b.append(int(init_b + init_b * (i / self.eta)))

        print("# Batch Sizes")
        for i in b:
            print(f"\t{i}")

        lr = [init_lr]
        for i in range(1, self.eta):
            lr.append(init_lr - init_lr * (i / self.eta))
            lr.append(init_lr + init_lr * (i / self.eta))

        print("# Learning Rates")
        for i in lr:
            print(f"\t{i}")

        hypers = []
        for nc_i in nc:
            for b_i in b:
                for lr_i in lr:
                    hypers.append([nc_i, b_i, lr_i])

        return hypers

    def run_trials(self, tr, va, nc, b, lr, epochs):
        final_scores = []
        conv_times = []
        epoch_scores = []
        # Run an average over multiple trials
        for i in range(self.num_trials):
            print(f" - Trial {i}/{self.num_trials}")
            # Get data and train the network
            nn = BPNN(nc, verbose=False)
            final_score, eval_scores, eval_scores_deltas, conv_time = nn.train(tr,
                                                                               max_epochs=epochs,
                                                                               batch_size=b,
                                                                               learning_rate=lr,
                                                                               evaluation_data=va,
                                                                               evaluation_treshold=None)

            final_scores.append(final_score)
            conv_times.append(conv_time)
            epoch_scores.append(eval_scores)

        # Compute the averages over the specified number of trials
        final_score_avg = np.average(final_scores)
        conv_times_avg = np.average(conv_times)

        epoch_scores = np.mean(epoch_scores, axis=0)

        return final_score_avg, epoch_scores, conv_times_avg

    def tune(self, init_nc, init_b, init_lr):
        """
        Tune the hyperparameters of the model using Hyperband tuning
        Train each model for increasingly long number of epochs, selecting the top models in each iteration to go on to the next
        """
        np.random.seed(0)

        epochs = 3
        best_hypers = [init_nc, init_b, init_lr]
        best_time = 0
        best_score = 0

        while epochs < self.max_epochs:
            print("=" * 80)
            print("# Epochs: ", epochs)

            hypers = self.generate_hypers(
                best_hypers[0], best_hypers[1], best_hypers[2])

            for i, hyper in enumerate(hypers):
                print()
                print("# Best Model: ", best_score)
                print(f" - Hypers: {best_hypers}")
                print(f" - Accuracy: {best_score}")
                print(f" - Time: {round(best_time, 2)}")

                print(f"# Current Model ({i}/{len(hypers)})")
                print(f" - Hypers: {hyper}")

                # Generate the training, validation and testing data
                tr, va, te, nnodes = BPNN.get_data()

                # Run the trials
                final_score, epoch_scores, conv_time = self.run_trials(
                    tr, va, hyper[0], hyper[1], hyper[2], epochs)

                # Update the best score
                if final_score > best_score:
                    best_score = final_score
                    best_time = conv_time
                    best_hypers = [hyper[0], hyper[1], hyper[2]]

                print(f" - Accuracy: {final_score}")
                print(f" - Time: {round(conv_time, 2)} sec")

            # Update the number of epochs
            epochs = int(epochs * self.eta)
