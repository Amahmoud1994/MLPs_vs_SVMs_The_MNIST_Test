from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from numba import jit

@jit(cache=True)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

@jit(cache=True)
def load_data():
    print("loading data")
    mnist = fetch_mldata("MNIST original")
    X, y = mnist.data / 255., mnist.target
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    images = X_train[0:20000]
    labels = y_train[0:20000]
    images_test = X_test[0:5000]
    labels_test = y_test[0:5000]
    return images, labels, images_test, labels_test


images, labels, images_test, labels_test = load_data()

# source anaconda3/bin/activate root
@jit(cache=True)
def mlp(images,labels,testImages,testLabels):

    mlp = MLPClassifier(solver='lbfgs',activation = 'relu',momentum = 0.9,
                     hidden_layer_sizes=(300,100), random_state=1,learning_rate  = 'constant',learning_rate_init  = 1,
                     early_stopping=False)

    mlp.fit(images, labels)
    # print(mlp.score(testImages,testLabels))

    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$$$$$$$$$ Start Plotting $$$$$$$$$$$$$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    title = "Learning Curves MLP"
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(mlp, title, images, labels, (0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()



@jit(cache=True)
def svc(images,labels,testImages,testLabels):
    svm = SVC(kernel = 'rbf', C = 10 , cache_size=2048, verbose=True)
    svm.fit(images, labels)
    # print(svm.score(testImages,testLabels))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    title = "Learning Curves SVC rbf,  C = 10 "
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    plot_learning_curve(svm, title, images, labels, (0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()



svc(images,labels, images_test,labels_test)
# mlp(images,labels, images_test,labels_test)
