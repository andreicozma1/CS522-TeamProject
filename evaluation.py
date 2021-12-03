import numpy as np
import matplotlib.pyplot as plt

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


def plot_confusion_matrix(mat, labels, title, filename:str=None):
    """
    plot a pretty confusion matrix given the matrix, the labels,
    the title of the plot, and optionally the filename if you want to save it
    """
    plt.title(title, fontsize=14, fontweight='bold')
    plt.imshow(mat, cmap='Blues', interpolation='nearest')
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    ax = plt.gca()

    # plot the corresponding number in the confusion matrix
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j,i,str(int(mat[i,j])),
                    fontsize=14,
                    color='black',
                    bbox={'facecolor':'white','alpha':1,'edgecolor':'black','pad':1.5},
                    ha='center', va='center')
    # remove ticks and fix layout
    ax.xaxis.tick_top()
    ax.axes.xaxis.set_ticks(labels)
    ax.axes.yaxis.set_ticks(labels)
    plt.colorbar()
    plt.tight_layout()

    # save to file it desired
    if filename is not None:
        plt.savefig(filename, dpi=500)
    
    plt.show()

def accuracy_score(y, y_model):
    """
    Return accuracy score.
    You are supposed to return both overall accuracy and classwise accuracy.
    The following code only returns overall accuracy
    """
    assert len(y) == len(y_model)

    classn = len(np.unique(y))  # number of different classes
    correct_all = y == y_model  # all correctly classified samples
    acc_overall = np.sum(correct_all) / len(y)
    acc_i = []  # this list stores classwise accuracy
    for i in range(classn):
        mask = np.ma.masked_where(y == i, y).mask  # create a mask of the current class
        acc_i.append(np.sum(correct_all, where=mask) / np.sum(mask))  # only sum correct values that are in the mask

    return acc_i, acc_overall