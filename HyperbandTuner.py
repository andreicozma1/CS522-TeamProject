from Dataset import *
from bpnn_cuda import BPNN
from tqdm import tqdm
import gzip
from evaluation import *
from model import Model
import pandas as pd


class HyperbandTuner:

    def __init__(self, model, num_trials=1, max_epochs=30, eta=3):
        self.model = model
        self.num_trials = num_trials
        self.max_epochs = max_epochs
        self.eta = eta

    def generate_hypers(self, init_nc, init_b, init_lr):
        print("# Generating Hyperparameters...")
        # Generate all combinations of hyperparameters
        nc = [init_nc]

        for i in range(1, self.eta):
            # don't change the first and last elements
            nc.append(
                [init_nc[0], int(init_nc[1] - init_nc[1] * (i / self.eta)), init_nc[-1]])
            nc.append(
                [init_nc[0], int(init_nc[1] + init_nc[1] * (i / self.eta)), init_nc[-1]])

        # print("\t- Network Configurations:")
        # for i in nc:
        #     print(f"\t\t{i}")

        b = [init_b]
        for i in range(1, self.eta):
            b.append(int(init_b - init_b * (i / self.eta)))
            b.append(int(init_b + init_b * (i / self.eta)))

        # print("\t - Batch Sizes:")
        # print(f"\t\t{b}")

        lr = [init_lr]
        for i in range(1, self.eta):
            lr.append(init_lr - init_lr * (i / self.eta))
            lr.append(init_lr + init_lr * (i / self.eta))

        # print("\t - Learning Rates:")
        # print(f"\t\t{lr}")

        # print out a table of possible network configurations, batch sizes, and learning rates on each column
        df = pd.DataFrame({'Network Configurations': nc,
                           'Batch Sizes': b, 'Learning Rates': lr})
        print(df.to_string(index=False))

        hypers = []
        for nc_i in nc:
            for b_i in b:
                for lr_i in lr:
                    hypers.append([nc_i, b_i, lr_i])

        return hypers

    def run_trials(self, nc, b, lr, epochs):
        final_scores = []
        conv_times = []
        epoch_scores = []
        best_acc = 0
        best_nn = BPNN(nc, verbose=False)

        # Run an average over multiple trials
        for i in range(self.num_trials):
            # print(f"# Trial {i + 1}/{self.num_trials}")
            # Get data and train the network
            tr, va, te, nnodes = BPNN.get_data()
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
            if final_score > best_acc:
                best_acc = final_score
                best_nn = nn

        # Compute the averages over the specified number of trials
        final_score_avg = np.average(final_scores)
        conv_times_avg = np.average(conv_times)

        epoch_scores = np.mean(epoch_scores, axis=0)

        return best_nn, final_score_avg, epoch_scores, conv_times_avg

    def evaluate(self, nn, data, name):
        y_test = []
        y_test_pred = []
        for x, y in data:
            decision = np.argmax(nn.feedforward(x))
            y_test.append(y.item())
            y_test_pred.append(decision.item())

        y_test = np.array(y_test)
        y_test_pred = np.array(y_test_pred)

        acc_i, acc_overall = accuracy_score(y_test, y_test_pred)

        # Print out the class 0, class 1, and overall accuracies along with the name

        return acc_i, acc_overall, y_test, y_test_pred

    def save_best_model(self, nn, y_test, y_test_pred):
        # Save the best BPNN model to a pickle file
        print("# Saving model")
        cm = get_confusion_matrix(y_test, y_test_pred, np.unique(y_test))
        Model(nn, y_test_pred, cm).save('bpnn.pkl')

    def tune(self, init_nc, init_b, init_lr):
        """
        Tune the hyperparameters of the model using Hyperband tuning
        Train each model for increasingly long number of epochs, selecting the top models in each iteration to go on to the next
        """
        np.random.seed(0)

        epochs = 3
        best_hypers = [init_nc, init_b, init_lr]

        best_overall_nn = BPNN(init_nc)
        best_overall_score = 0
        best_overall_epoch_scores = []
        best_overall_time = 0

        while epochs < self.max_epochs:
            print("=" * 80)
            print("# Epochs: ", epochs)

            hypers = self.generate_hypers(
                best_hypers[0], best_hypers[1], best_hypers[2])

            for hyper in hypers:

                # Generate the training, validation and testing data
                tr, va, te, nnodes = BPNN.get_data()

                # Run the trials
                best_nn, final_score, epoch_scores, conv_time = self.run_trials(
                    hyper[0], hyper[1], hyper[2], epochs)

                # Update the best score
                if final_score > best_overall_score:
                    best_overall_nn = best_nn
                    best_overall_score = final_score
                    best_overall_epoch_scores = epoch_scores
                    best_overall_time = conv_time
                    best_hypers = [hyper[0], hyper[1], hyper[2]]
                    print("-" * 60)
                    # In a pandas dataframe with columns:
                    # Network Configuration, Batch Size, Learning Rate, Epochs, Final Accuracy, Training Time

                    print("# Found New Best Model: ", best_overall_score)
                    df = pd.DataFrame(
                        {'': ['Model Info'],
                         'Layers': [hyper[0]],
                         'Batch Size': [hyper[1]],
                         'Learning Rate': [hyper[2]],
                         'Epochs': [epochs],
                         'Val Acc': [best_overall_score],
                         'Time': [best_overall_time]})
                    print(df.to_string(index=False))

                    tr, va, te, nnodes = BPNN.get_data()

                    acc_val_i, acc_val_overall, y_val, y_val_pred = self.evaluate(
                        best_overall_nn, va, 'Validation')
                    acc_test_i, acc_test_overall, y_test, y_test_pred = self.evaluate(
                        best_overall_nn, te, 'Testing')

                    # Columns:
                    # - validation and testing print
                    # Rows: class 0, class 1, and overall accuracy

                    df = pd.DataFrame(
                        {'Class 0': [acc_val_i[0], acc_test_i[0]],
                         'Class 1': [acc_val_i[1], acc_test_i[1]],
                         'Overall': [acc_val_overall, acc_test_overall]},
                        index=['Validation', "Testing"])
                    print(df.to_string(index=True))

                    self.save_best_model(best_overall_nn, y_test, y_test_pred)

            # Update the number of epochs
            epochs = int(epochs * self.eta)

        print("# Best Overall Model: ", best_overall_score)
        df = pd.DataFrame(
            {'': ['Model Info'],
                'Layers': [best_hypers[0]],
                'Batch Size': [best_hypers[1]],
                'Learning Rate': [best_hypers[2]],
                'Epochs': [epochs],
                'Val Acc': [best_overall_score],
                'Time': [best_overall_time]})
        return best_overall_nn, best_overall_score, best_overall_epoch_scores, best_overall_time
