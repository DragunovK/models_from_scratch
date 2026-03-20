from ..autograd import Node, topo_sort
from .layer import Layer, Dropout
from ..optim.optimizer import Optimizer
from ..nn.loss import Loss

import numpy as np
from collections import defaultdict


class Model:
    def params(self):
        return []

    def __call__(self, x):
        raise NotImplementedError("Abstract Model does not implement __call__")


class Sequential(Model):
    def __init__(self, layers: list[Layer]):
        self._layers = layers

        self._loss = Loss()
        self._optimizer = Optimizer()

        self._eval_mode = False

    def __call__(self, x: Node):
        """Forward pass"""

        if not self._eval_mode:
            for layer in self._layers:
                x = layer(x)
            return x

        # if in eval mode (validation / actual predictions)
        # skip the dropout layers
        for layer in self._layers:
            if isinstance(layer, Dropout):
                continue
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

    def compile_(self, optimizer: Optimizer, loss: Loss):
        self._optimizer = optimizer
        self._loss = loss
        self._params = self._get_params()

    def __fb(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Performs one-batch forward and then backward pass.
        Returns loss.
        """
        x_node = Node(X, node_type="Input")
        y_est = self(x_node)

        loss_node, loss_val = self._loss(y_est, Y)
        self.backward(loss_node)

        return loss_val

    def __fb_noback(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Version of __fb(X, Y) that computes loss but does not
        compute gradients (backward pass is not performed)
        """
        x_node = Node(X, node_type="Input")

        self._eval_mode = True  # enable eval mode to skip dropouts
        y_est = self(x_node)
        self._eval_mode = False

        _, loss_val = self._loss(y_est, Y)
        return loss_val

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int,
        batch_size: int = 1,  # 0 < batch_size <= n
        validation_split: float = 0.0,  # 0 <= validation_split <= 0.9
        shuffle: bool = True,
        shuffle_before_split: bool = True,
        verbose: bool = True,
        rich_history: bool = False,
    ) -> defaultdict[str, list]:
        """Fits model to provided data. This function runs the main learning loop.
        Params:
            X(np.ndarray): image samples w/ shape (n_samples, image_len)
            Y(np.ndarray): label samples w/ shape (n_samples, label_len)
            epochs(integer > 0): number of epochs to run training loop for
            batch_size(integer in range [1, n_samples]): batch size
            validation_split(float in range [0, 0.9]): validation split
            shuffle(bool): whether to shuffle the training data each epoch or not
            shuffle_before_split(bool): whether to shuffle X before train:validation split
            verbose(bool): whether to print performance stats each epoch
            rich_history(bool): whether to include momentums and learning rates in returned hist. (TODO)
        Returns:
            history(dict): training history
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        n_samples = X.shape[0]

        if validation_split:
            idx = np.arange(n_samples)
            if shuffle_before_split:
                np.random.shuffle(idx)

            split_point = int(validation_split * n_samples)
            n_samples = split_point  # batch slicing is based on n_samples

            train_idx = idx[split_point:]
            valid_idx = idx[:split_point]

            X_train = X[train_idx]
            X_valid = X[valid_idx]

            Y_train = Y[train_idx]
            Y_valid = Y[valid_idx]
        else:
            X_train = X
            Y_train = Y

        self.history = defaultdict(list)
        self._eval_mode = False

        for epoch in range(epochs):
            idx = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(idx)

            total_loss = 0.0
            if batch_size == 1:
                total_loss = self.__fb(X_train[idx], Y_train[idx])
                self._optimizer.step(self._params)
            else:
                for batch_left in range(0, n_samples, batch_size):
                    batch_idxs = idx[batch_left : (batch_left + batch_size)]
                    x_batch = X_train[batch_idxs]
                    y_batch = Y_train[batch_idxs]

                    batch_loss = self.__fb(x_batch, y_batch)
                    self._optimizer.step(self._params)

                    total_loss += batch_loss

            avg_loss = total_loss / batch_size
            self.history["train_loss"].append(avg_loss)

            if verbose:
                print(f"[INFO] Epoch{epoch}: train_loss={avg_loss:.6f}")

            if validation_split:
                valid_loss = self.__fb_noback(X_valid, Y_valid)
                self.history["valid_loss"].append(valid_loss)

                if verbose:
                    print(f"\t\t valid_loss={valid_loss:.6f}")

        self._eval_mode = True  # training is done, enable eval mode
        return self.history
