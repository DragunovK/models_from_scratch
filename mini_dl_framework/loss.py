from autograd import Node

import numpy as np


class Loss:
    def __call__(self, y_est: Node, y_true: np.ndarray) -> tuple[Node, float]:
        raise NotImplementedError("Abstract Loss does not implement '__call__'.")


class SumOfSquares(Loss):
    def __call__(self, y_est: Node, y_true: np.ndarray):
        y_true = np.asarray(y_true, dtype=float)
        diff = y_est.value - y_true
        N = y_est.value.shape[0]

        loss_val = 0.5 * np.sum(diff**2) / N
        loss_node = Node(np.array(loss_val), (y_est,), "SumSquares")

        def __grad():
            y_est.gradient = y_est.gradient + (diff / N) * loss_node.gradient

        # IMPORTANT: name-mangled attribute
        loss_node._Node__grad = __grad

        return loss_node, float(loss_val)
