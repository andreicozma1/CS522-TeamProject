from Dataset import *
import cupy as cp
from bpnn_cuda import BPNN


def convert_categories(y_cat):
    desired = cp.array([0 if y[0] == 1 else 1 for y in y_cat])
    return desired


def get_data():
    d = Dataset.load_gzip(os.path.join(
        "datasets", "face_mask_pickled"), "dataset_gray_conv.pkl.gzip")

    ndimen = d.train.X.shape[1]

    # Structure Training Data for BPNN
    training_inputs = [cp.reshape(cp.asarray(x), (ndimen, 1))
                       for x in d.train.X]
    training_results = [cp.asarray(y.reshape(-1, 1)) for y in d.train.y]

    training_data = zip(training_inputs, training_results)
    # Structure Validation Data for BPNN
    validation_inputs = [cp.reshape(cp.asarray(x), (ndimen, 1))
                         for x in d.validation.X]
    validation_data = zip(
        validation_inputs, convert_categories(d.validation.y))
    # Structure Testing Data for BPNN
    testing_inputs = [cp.reshape(cp.asarray(x), (ndimen, 1)) for x in d.test.X]
    testing_data = zip(testing_inputs, convert_categories(d.test.y))

    return training_data, validation_data, testing_data, ndimen


ntrials = 1
max_epochs = 5


# model_BPNN = BPNN(init_nc, verbose=True)
# final_score, eval_scores, eval_scores_deltas, conv_time = model_BPNN.train(training_data,
#                                                                            max_epochs=max_epochs,
#                                                                            batch_size=init_b,
#                                                                            learning_rate=init_lr,
#                                                                            evaluation_data=validation_data)

def run_hyperband_tuning():
    training_data, validation_data, testing_data, ndimen = get_data()
    print(f"Input Dimension: {ndimen}")

    ht = HyperbandTuner(BPNN, max_epochs=max_epochs, eta=3)

    print(ht.generate_hypers([ndimen, 500, 250, 50, 2], 20, 0.35))

    ht.tune(training_data, validation_data, testing_data)

    # print(ht.get_best_hyperparameters())


class HyperbandTuner:

    def __init__(self, model, max_epochs=30, eta=3, seed=None):
        self.model = model
        self.max_epochs = max_epochs
        self.eta = eta
        if seed is not None:
            self.seed = seed
        else:
            np.random.seed(self.seed)

        self.hypers = []
        self.best_hypers = []

    def generate_hypers(self, init_nc, init_b, init_lr):
        # Generate all combinations of hyperparameters
        nc = [init_nc[0]]
        for i in range(1, len(init_nc)):
            nc.append(init_nc[i])
            for j in range(1, self.eta):
                nc.append(init_nc[i] + j * (init_nc[i] - init_nc[i - 1]))
        b = [init_b]
        for i in range(1, self.eta):
            b.append(init_b + i * (init_b - 1))
        lr = [init_lr]
        for i in range(1, self.eta):
            lr.append(init_lr * (1 + (i / self.eta)))

        hypers = []
        for nc_i in nc:
            for b_i in b:
                for lr_i in lr:
                    hypers.append([nc_i, b_i, lr_i])

        self.hypers = hypers
        return hypers

    def get_hypers(self):
        return self.hypers

    def get_best_hypers(self):
        return self.best_hypers

    def tune(self, training_data, validation_data, testing_data, seed=None):
        # run trials
        for trial in range(ntrials):
            print(f"Trial {trial}")
            # shuffle hyperparameters
            np.random.shuffle(self.hypers)

            # keep track of best hyperparameters
            best_score = -1
            best_hyper = []

            # run hyperparameter trials
            for hyper in self.hypers:
                print(f"# Hyperparameters:")
                print(f'\t- NC: {hyper[0]}')
                print(f'\t- B: {hyper[1]}')
                print(f'\t- LR: {hyper[2]}')
                dt = np.copy(training_data)
                dv = np.copy(validation_data)

                model_BPNN = self.model(hyper[0], verbose=True)
                score, eval_scores, eval_scores_deltas, conv_time = model_BPNN.train(
                    training_data, max_epochs=self.max_epochs, batch_size=hyper[
                        1], learning_rate=hyper[2],
                    dt=dv)

                # keep track of best hyperparameters
                if score > best_score:
                    best_score = score
                    best_hyper = hyper

            # store best hyperparameters
            self.best_hypers.append(best_hyper)
            print()
            print('=' * 80)
            print(f"# Best Overall: {best_score}")
            print(f'\t- NC: {best_hyper[0]}')
            print(f'\t- B: {best_hyper[1]}')
            print(f'\t- LR: {best_hyper[2]}')

        # print best hyperparameters
        for hyper in self.best_hypers:
            print(f"Best Hyperparameters: {hyper}")
