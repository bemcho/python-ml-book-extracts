import numpy as np


def input_std(X: np.ndarray, idxList: list = None):
    """Standardization shifts the mean of each
    feature so that it is centered at zero and each feature has a standard deviation of 1.

    formulae:
        x ′ j = (x j − μ j) / σ j

    :param X: ndarray to be processed
    :param idxList: list of indexes in X[:,idxList] to be normalized if None all are processed

    :return: same shape as X but with normalized data in X.shape[1]

    """
    X_std = np.copy(X)
    for i in range(X.shape[1]) if idxList is None else idxList:
        X_std[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    return X_std
