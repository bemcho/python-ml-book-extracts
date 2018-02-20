# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ## An object-oriented perceptron API
from notes.AdalineGD import AdalineGD
from notes.AdalineSGD import AdalineSGD
from notes.Perceptron import Perceptron
from notes.PlotUtils import plot_decision_regions, plot_epoch_progress, plot_test_set_prediction
# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
#
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
#
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)
# # Python Machine Learning - Code Examples
# # Chapter 2 - Training Machine Learning Algorithms for Classification
# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).
# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*
# ### Overview
#
# - [Artificial neurons – a brief glimpse into the early history of machine learning](#Artificial-neurons-a-brief-glimpse-into-the-early-history-of-machine-learning)
#     - [The formal definition of an artificial neuron](#The-formal-definition-of-an-artificial-neuron)
#     - [The perceptron learning rule](#The-perceptron-learning-rule)
# - [Implementing a perceptron learning algorithm in Python](#Implementing-a-perceptron-learning-algorithm-in-Python)
#     - [An object-oriented perceptron API](#An-object-oriented-perceptron-API)
#     - [Training a perceptron model on the Iris dataset](#Training-a-perceptron-model-on-the-Iris-dataset)
# - [Adaptive linear neurons and the convergence of learning](#Adaptive-linear-neurons-and-the-convergence-of-learning)
#     - [Minimizing cost functions with gradient descent](#Minimizing-cost-functions-with-gradient-descent)
#     - [Implementing an Adaptive Linear Neuron in Python](#Implementing-an-Adaptive-Linear-Neuron-in-Python)
#     - [Improving gradient descent through feature scaling](#Improving-gradient-descent-through-feature-scaling)
#     - [Large scale machine learning and stochastic gradient descent](#Large-scale-machine-learning-and-stochastic-gradient-descent)
# - [Summary](#Summary)
# # Artificial neurons - a brief glimpse into the early history of machine learning
# ## The formal definition of an artificial neuron
# ## The perceptron learning rule
# # Implementing a perceptron learning algorithm in Python
from notes.Utils import input_std

# ### Reading-in the Iris data
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)
df.tail()

#
# ### Note:
# 
# 
# You can find a copy of the Iris dataset (and all other datasets used in this book) in the code bundle of this book, which you can use if you are working offline or the UCI server at https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data is temporarily unavailable. For instance, to load the Iris dataset from a local directory, you can replace the line 
# 
#     df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#         'machine-learning-databases/iris/iris.data', header=None)
#  
# by
#  
#     df = pd.read_csv('your/local/path/to/iris.data', header=None)
# 


#df = pd.read_csv('iris.data', header=None)
#df.tail()

# ### Plotting the Iris data


# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_06.png', dpi=300)
plt.show()

# ### Training the perceptron model


ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plot_epoch_progress(ppn.errors_)
plot_test_set_prediction(X, y, ppn, 'sepal length [cm]', 'petal length [cm]', title='Perceptron')

# # Adaptive linear neurons and the convergence of learning

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
plot_epoch_progress(ada1.cost_, y_label='log(Sum-squared-error)', title='Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
plot_epoch_progress(ada2.cost_, y_label='Sum-squared-error', title='Adaline - Learning rate 0.0001')

# ## Improving gradient descent through feature scaling


# standardize features
X_std = input_std(X, [0, 1])

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plot_epoch_progress(ada.cost_, y_label='Sum-squared-error', title='Adaline - Gradient Descent')
plot_test_set_prediction(X_std, y, ada, x_label='sepal length [standardized]', y_label='petal length [standardized]', title='Adaline - Gradient Descent')

# ## Large scale machine learning and stochastic gradient descent


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_epoch_progress(ada.cost_, y_label='Average Cost', title='Adaline - Stochastic Gradient Descent')
plot_test_set_prediction(X_std, y, ada, x_label='sepal length [standardized]', y_label='petal length [standardized]', title='Adaline - Stochastic Gradient Descent')

ada.partial_fit(X_std[0, :], y[0])
