from autograd import Node

import numpy as np

"""
How to create a Layer with a custom operation, not default Node ops:

class CustomLayer(Layer):
    def forward(self, x: Node) -> Node:
        result_val = f(x.value)

        result = Node(result_val, (x,), "CustomLayer")
        def __grad():
            x.gradient = x.gradient + ...

        # IMPORTANT: name-mangled attribute
        result._Node__grad = __grad

        return result
"""


class Layer:
    def params(self):
        return []

    def __call__(self, x: Node) -> Node:
        return self.forward(x)

    def forward(self, x: Node) -> Node:
        raise NotImplementedError("Abstract Layer does not implement method 'forward'.")


class Dense(Layer):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator | None = None, init_type: str = "he"):
        rng = rng or np.random.default_rng()

        init_type = init_type.lower()
        match init_type:
            case "he":
                w_val = rng.normal(0.0, 2 / n_in, size=(n_in, n_out))
            case "std" | "std_norm" | "std_gauss":
                w_val = rng.normal(0.0, 1.0, size=(n_in, n_out))
            case _:
                raise ValueError(f"Unknown init type: {init_type}. Options: he, (std_norm | std_gauss)")

        self.w = Node(w_val, node_type="W")
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
