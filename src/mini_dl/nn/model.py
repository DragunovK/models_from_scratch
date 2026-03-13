from ..autograd import Node, topo_sort
from .layer import Layer

import numpy as np


class Model:
    def params(self):
        return []

    def __call__(self, x):
        raise NotImplementedError("Abstract Model does not implement __call__")


class Sequential(Model):
    def __init__(self, layers: list[Layer]):
        self._layers = layers
        self._loss = None
        self._optimizer = None

    def __call__(self, x: Node):
        """Forward pass"""

        for layer in self._layers:
            x = layer(x)
        return x

    def backward(self, loss_node: Node):
        topo_sort(loss_node)
        loss_node.reset_grad()
        loss_node.propagate_back(np.array(1.0))

    def _get_params(self) -> list[Node]:
        result = []
        for layer in self._layers:
            p = layer.params()
            result.extend(p)
        return result

    def compile_(self, optimizer, loss):
        self._optimizer = optimizer
        self._loss = loss
        self._params = self._get_params()

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int,
        batch_size: int = -1,
        validation_split: int = 0,
        shuffle: bool = True,
        verbose: bool = True,
    ):
        X = np.asarray(X)
        Y = np.asarray(Y)

        n = X.shape[0]
        self.history = {"loss": []}

        for epoch in range(epochs):
            idx = np.arange(n)
            if shuffle:
                np.random.shuffle(idx)

            total_loss = 0.0
            n_batches = 0

            if batch_size <= 0:
                x_batch = X[idx]
                y_batch = Y[idx]

                x_batch_node = Node(x_batch, node_type="Input")
                y_est = self(x_batch_node)

                loss_node, loss_val = self._loss(y_est, y_batch)
                self.backward(loss_node)

                self._optimizer.step(self._params)
                total_loss = float(loss_val)
            else:
                for batch_left in range(0, n, batch_size):
                    batch_idxs = idx[batch_left : (batch_left + batch_size)]
                    x_batch = X[batch_idxs]
                    y_batch = Y[batch_idxs]

                    x_batch_node = Node(x_batch, node_type="Input")
                    logits = self(x_batch_node)

                    batch_loss_node, batch_loss_val = self._loss(logits, y_batch)
                    self.backward(batch_loss_node)

                    self._optimizer.step(self._params)

                    total_loss += float(batch_loss_val)
                    n_batches += 1

            avg_loss = total_loss / (n_batches or 1)
            self.history["loss"].append(avg_loss)

            if verbose:
                print(f"Epoch{epoch}: loss={avg_loss:.6f}")

        return self.history
