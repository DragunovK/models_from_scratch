from ..autograd import Node

import numpy as np


class Loss:
    def __call__(self, y_est: Node, y_true: np.ndarray) -> tuple[Node, float]:
        raise NotImplementedError("Abstract Loss does not implement '__call__'.")


class SumOfSquares(Loss):
    def __call__(self, y_est: Node, y_true: np.ndarray) -> tuple[Node, float]:
        y_true = np.asarray(y_true, dtype=np.float64)
        diff = y_est.value - y_true
        N = y_est.value.shape[0]

        loss_val = 0.5 * np.sum(diff**2) / N
        loss_node = Node(loss_val, (y_est,), "SumSquares")

        def __grad():
            y_est.gradient = y_est.gradient + (diff / N) * loss_node.gradient

        # IMPORTANT: name-mangled attribute
        loss_node._Node__grad = __grad  # type: ignore[attr-defined]
        return loss_node, float(loss_val)


class CrossEntropy(Loss):
    def __call__(self, y_est: Node, y_true: np.ndarray) -> tuple[Node, float]:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_est_val = y_est.value

        # Softmax on the y_est to -> it into a valid prob. distribution
        y_est_val = y_est_val - np.max(y_est_val, axis=1, keepdims=True)
        exponent = np.exp(y_est_val)
        y_est_val = exponent / np.sum(exponent, axis=1, keepdims=True)

        N = y_est_val.shape[0]

        # cross-entropy
        loss_val = -np.sum(y_true * np.log(y_est_val)) / N
        loss_node = Node(loss_val, (y_est,), "CrossEntropy")

        def __grad():
            y_est.gradient = y_est.gradient + ((y_est_val - y_true) / N) * loss_node.gradient

        loss_node._Node__grad = __grad  # type: ignore[attr-defined]
        return loss_node, float(loss_val)
