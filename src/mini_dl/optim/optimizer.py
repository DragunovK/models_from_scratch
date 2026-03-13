from ..autograd import Node

import numpy as np


class Optimizer:
    def step(self, params):
        raise NotImplementedError("Abstract Optimizer does not implement 'step'.")


class SGDOptimizer(Optimizer):
    """
    UDL 6.2 Stochastic gradient descent

    Implementation follows the logic of:
        https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/src/optim/sgd.cpp
    except for the decay and dampening, which are to be added later.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0, nesterov: bool = False):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.momentum_cache: dict[int, np.ndarray] = {}

    def step(self, params: list[Node]):
        for p in params:
            if self.momentum > 0.0:
                m_prev = self.momentum_cache.get(id(p))
                if m_prev is None:
                    m_prev = np.zeros_like(p.value)

                m = self.momentum * m_prev + p.gradient
                self.momentum_cache[id(p)] = m

                if self.nesterov:
                    p.gradient = p.gradient + self.momentum * m
                else:
                    p.gradient = m

            p.value = p.value - self.lr * p.gradient


class Adam(Optimizer):
    """
    UDL 6.4 Adam page 90
    """

    def __init__(self, lr: float = 0.001, beta: float = 0.9, gamma: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

        self.m: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}

        self.t = 0

    def step(self, params: list[Node]):
        self.t += 1
        for p in params:
            m_prev = self.m.get(id(p))
            if m_prev is None:
                m_prev = np.zeros_like(p.value)

            m = self.beta * m_prev + (1 - self.beta) * p.gradient
            self.m[id(p)] = m

            v_prev = self.v.get(id(p))
            if v_prev is None:
                v_prev = np.zeros_like(p.value)

            v = self.gamma * v_prev + (1 - self.gamma) * (p.gradient**2)
            self.v[id(p)] = v

            m_tilde = m / (1 - self.beta**self.t)
            v_tilde = v / (1 - self.gamma**self.t)

            p.value = p.value - self.lr * m_tilde / (np.sqrt(v_tilde) + self.eps)
