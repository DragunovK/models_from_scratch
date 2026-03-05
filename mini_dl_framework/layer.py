from autograd import Node

import numpy as np


class Layer:
    def params(self):
        return []

    def __call__(self, x: Node) -> Node:
        return self.forward(x)

    def forward(self, x: Node) -> Node:
        raise NotImplementedError("Abstract Layer does not implement method 'forward'.")


class Dense(Layer):
    def __init__(self, n_in, n_out, rng=None, he: bool = True):
        rng = rng or np.random.default_rng()

        if he:
            self.w = Node(rng.normal(0.0, 2 / n_in, size=(n_in, n_out)), node_type="W")
        else:
            self.w = Node(rng.normal(0.0, 1.0, size=(n_in, n_out)), node_type="W")

        self.b = Node(np.zeros((n_out,), dtype=np.float32), node_type="B")

    def forward(self, x: Node) -> Node:
        return (x @ self.w) + self.b

    def params(self) -> list[Node]:
        return [self.w, self.b]


class ReLU(Layer):
    def __init__(self, clip: int = 0):
        self.clip = clip

    def forward(self, x: Node) -> Node:
        if self.clip:
            x.value = np.minimum(x.value, self.clip)
        return x.relu()


class SoftMax(Layer):
    def forward(self, x: Node) -> Node:
        return x.softmax()
