from autograd import Node

import numpy as np


class Optimizer:
    def step(self, params):
        raise NotImplementedError("Abstract Optimizer does not implement 'step'.")


class SGDOptimizer(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.0, nesterov: bool = False):
        self.lr = lr
        self.momentum = momentum
        self.momentum_cache: dict[int, np.ndarray] = {}

    def step(self, params: list[Node]):
        for p in params:
            if self.momentum > 0.0:
                m_prev = self.momentum_cache.get(id(p))
                if m_prev is None:
                    m_prev = np.zeros_like(p.value)

                m = self.momentum * m_prev - self.lr * p.gradient
                p.value = p.value + m
                self.momentum_cache[id(p)] = m
            else:
                p.value = p.value - self.lr * p.gradient


class Adam(Optimizer):
    def step(self, params):
        pass
