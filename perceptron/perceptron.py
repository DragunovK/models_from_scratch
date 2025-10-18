""" Classic perceptron implementation. 
Signature is based on: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
Algorithm description: Learning from Data page 7
"""

import numpy as np


class Perceptron:
    def __init__(self, max_iter: int, tol: float, verbose: bool, eta0: float):
        """ 
        Args:
            max_iter (int): upper limit to the number of training epochs
            tol (float): stopping criterion
            verbose (bool): whether or not to print training information
            eta0 (float): initial learning rate
        """
        self.max_iter = max_iter
        
        self.tol = tol
        self.eta0 = eta0

        self.verbose = verbose

        self.B = np.random.randn()

        def sign(x):
            return np.ones_like(x) if x > 0 else np.zeros_like(x)
        self.activation = sign


    def fit(self, X, Y):
        """ Fit perceptron to provided data.
        """
        self.data_len = X.shape[0]
        X = X.reshape(*X.shape, 1)

        self.input_shape = X[0].shape
        self.output_shape = Y[0].shape

        self.W = np.random.randn(self.input_shape[0], self.output_shape[0])

        self.mse = 1e6
        self.epoch = 0
        while self.epoch < self.max_iter and self.mse > self.tol:
            self.epoch += 1

            for Xi, Yi in zip(X, Y):
                Y_hat = self.activation(self.W.T @ Xi + self.B)
                self.W += self.eta0 * (Yi - Y_hat) * Xi
                self.B += self.eta0 * (Yi - Y_hat)
            
            # eval
            total_error = 0
            for Xi, Yi in zip(X, Y):
                Y_hat = self.activation(self.W.T @ Xi + self.B)
                total_error += (Yi - Y_hat) ** 2
                
            self.mse = total_error / self.data_len

            if self.verbose:
                print(f'Epoch {self.epoch}| MSE {self.mse}')

    def predict(self, x):
        return self.activation(self.W.T @ x + self.B)

    @property
    def summary(self):
        return {
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'epoch': self.epoch,
            'loss': self.mse,
            'eta0:': self.eta0,
            'W': self.W,
            'B': self.B
        }


if __name__ == '__main__':
    DATA = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])

    X, Y = DATA[:, :2], DATA[:, 2:]
    
    perceptron = Perceptron(10000, 1e-3, True, 0.01)

    perceptron.fit(X, Y)

    print([0, 0], perceptron.predict(np.array([0, 0])))
    print([0, 1], perceptron.predict(np.array([0, 1])))
    print([1, 0], perceptron.predict(np.array([1, 0])))
    print([1, 1], perceptron.predict(np.array([1, 1])))

    print(perceptron.summary)
