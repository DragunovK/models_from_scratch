from sklearn.datasets import fetch_openml

import numpy as np


def as_one_hot(arr: np.ndarray):
    return np.eye(np.max(arr + 1))[arr]


def get_mnist_dataset(train_split=0.7):
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, cache=True)

    X, Y = mnist.data, mnist.target.astype(int)

    X = X.astype(np.float32) / 255.0
    Y = as_one_hot(Y).astype(np.float32)

    indices = np.random.permutation(X.shape[0])
    split_point = int(X.shape[0] * train_split)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

    return X_train, Y_train, X_test, Y_test
