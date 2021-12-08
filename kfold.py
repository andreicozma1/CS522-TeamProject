import numpy as np
from itertools import product
from multiprocessing import Pool


# get a list of the indices to be used for the training set with the given k
def gen_train_idx(K, k, n):
    return np.array([i for i in range(n) if i % K != k])


# get the cartesian product of all the list values of the keys
def dict_cartesian_product(d):
    k, V = zip(*[(k,v) for k,v in d.items()])
    prod = list(product(*V))
    return [dict(zip(k,v)) for v in prod]

def fold(model, K, hypers, X, y):
    n = len(y)
    acc = []

    # train k models and evaluate
    for k in range(K):
        # split the data into train and test sections
        idx = gen_train_idx(K, k, n)
        train_X, train_y = X[idx], y[idx]
        test_X, test_y = X[~idx], y[~idx]

        # train model with current set of hypers
        m = model(**hypers)
        m.fit(train_X, train_y)
        y_pred = m.predict(test_X)

        # calculate and save accuracy
        cur_acc = np.sum(test_y == y_pred) / len(y_pred)
        acc.append(cur_acc)

    # get average accuracy
    avg_acc = sum(acc) / len(acc)
    return (avg_acc, hypers)

def Kfold(K: int, model, X, y, hyperparam_grid, n_workers=1, verbose=False, ret_all_accs=False):
    """
    :param K: number of folds
    :param model: pass the reference to the models class. must have fit and predict functions
    :param X: train and validation x
    :param y: train and validation y
    :param hyperparam_grid: matrix of desired hyperparameters
    :param n_workers: number of processes to train on
    :param verbose: print out results
    :param ret_all_accs: return all the accuracies and hyperparameter configurations
    :return: best accuracy and best parameters
    """
    assert K >= 2
    assert len(X) == len(y)

    # generate list of hyperparameters
    hyper_list = [hypers for hyper_dict in hyperparam_grid for hypers in dict_cartesian_product(hyper_dict)]

    # generate list of parameters to fold function
    fold_param_list = [(model, K, h, X, y) for h in hyper_list]

    # spawn multiple fold processes
    with Pool(processes=n_workers) as pool:
        result = pool.starmap_async(fold, fold_param_list)
        acc_param = result.get(timeout=None)

    # get the best accuracy and parameters
    accs, params = zip(*acc_param)
    best_acc, best_params = get_best_acc(accs, params)

    # if is verbose
    if verbose:
        # print all tried hyperparameters with the corresponding accuracies
        print(f'#' * 100)
        for a, p in zip(accs, params):
            print(f'### hyperparameter:', p)
            print(f'### accuracy:',a)
            if a != accs[-1]:
                print(f'-' * 100)

        # print the best parameters and accuracy
        print(f'#' * 100)
        print(f'### best hyperparameters:', best_params)
        print(f'### best accuracy:', best_acc)
        print(f'#' * 100)

    if ret_all_accs:
        return accs, params
    else:
        return best_acc, best_params

def get_best_acc(accs, params):
    best_idx = np.argmax(np.array(accs))
    best_acc, best_params = accs[best_idx], params[best_idx]
    return best_acc, best_params