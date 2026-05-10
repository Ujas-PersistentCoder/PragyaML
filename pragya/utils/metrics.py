import numpy as np

"""=============================================================================="""

def entropy(arr):

    """Calculates the entropy of a label/class array"""

    if len(arr) == 0: return 0
    _, counts = np.unique(arr, return_counts = True)
    probabilities = counts / len(arr)
    return -np.sum(probabilities * np.log2(probabilities))


def gini(arr): #faster to calculate than entropy

    """Calculates the gini impurity of a label/class array"""

    if len(arr) == 0: return 0
    _, counts = np.unique(arr, return_counts = True)
    probabilities = counts / len(arr)
    return 1 - np.sum(probabilities ** 2)

def mean_squared_error(arr):

    """Calculates the mean squared error of a numerical array"""

    if len(arr) == 0: return 0
    mean = np.mean(arr)
    mse = np.mean((arr - mean) ** 2)
    return mse


def gain(parent, left_child, right_child, metric):

    """Calculates the gain due to a split taking in the input arrays of the parent and the left and right child
       and a parameter to specify the metric to use (entropy or gini)"""
    if metric == None:
        raise ValueError("Please specify a metric to calculate gain")
    if metric == "entropy":
        parent_metric = entropy(parent)
        left_metric = entropy(left_child)
        right_metric = entropy(right_child)
    elif metric == "gini":
        parent_metric = gini(parent)
        left_metric = gini(left_child)
        right_metric = gini(right_child)
    elif metric == "mse":
        parent_metric = mean_squared_error(parent)
        left_metric = mean_squared_error(left_child)
        right_metric = mean_squared_error(right_child)
    else:
        raise ValueError("Invalid metric. Valid metrics: entropy, gini, or mse")
    total = len(parent)
    gain_value = parent_metric - (len(left_child) / total) * left_metric - (len(right_child) / total) * right_metric
    return gain_value