import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


def plot_epoch_progress(errors_or_cost: np.ndarray, x_label: str = 'Epochs', y_label: str = 'Number of updates',
                        title: str = 'Error/Cost progress per Epoch'):
    """Plots Epoch as X axis and number of updates as Y axis"""
    plt.plot(range(1, len(errors_or_cost) + 1), errors_or_cost, marker='o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_test_set_prediction(predicted: np.ndarray, expected: np.ndarray, alg, x_label: str = 'Property X',
                             y_label: str = 'Property Y',
                             title: str = 'NN name', legend_location: str = 'upper left'):
    plot_decision_regions(predicted, expected, classifier=alg)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_location)
    plt.tight_layout()
    plt.show()
