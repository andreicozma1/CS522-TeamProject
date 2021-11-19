import numpy as np

# example usage: get_confusion_matrix(y, y_pred, np.unique(y))
def get_confusion_matrix(y, y_model, label_order):
    """
    get confusion matrix given the actual y, the predicted y, and the order
    the user wants the labels to be in, in the confusion matrix
    y axis is ground truth labels
    x axis is predicted labels
    """
    label_order = np.array(label_order)
    mat = np.zeros((len(label_order),len(label_order)))

    # get mask of all correct labels
    correct_all = y == y_model
    for i, l in enumerate(label_order):
        # create mask where actual labels are equal to the current label
        m1 = np.ma.masked_where(y == l, y).mask

        # create mask where predicted labels are not equal to current label
        m2 = np.ma.masked_where(y_model != l, y).mask

        # sum all correct labels that correspond to the current label
        mat[i,i] = np.sum(correct_all, where=m1)

        # get list of all incorrect prediction for the current label
        incorrect = np.array(y_model)[(m1 == True) & (m2 == True)]

        # find the unique labels and the counts in the set of incorrect predictions
        incorrect = np.unique(incorrect, return_counts=True)

        # add the incorrect counts to their corresponding locations in the matrix
        for i_l, c in zip(incorrect[0],incorrect[1]):
            res = np.where(label_order==i_l)[0][0]
            mat[i,res] = c
    return np.array(mat)
