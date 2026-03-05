import numpy as np


class Node:
    def __init__(self, value: np.ndarray, children: tuple["Node", ...] = (), node_type: str = "V"):
        self.value = value
        self.gradient = np.zeros_like(value)
        self.children = children
        self.visited = False
        self.node_type = node_type
        self.__grad = lambda: None
        self.nodes: list["Node"] | None = None

    def __matmul__(self, other: "Node") -> "Node":
        other = other if isinstance(other, Node) else Node(other)
        result = Node(self.value @ other.value, (self, other), "@")

        def __grad():
            self.gradient = self.gradient + (result.gradient @ other.value.T)
            other.gradient = other.gradient + (self.value.T @ result.gradient)

        result.__grad = __grad

        return result

    def __add__(self, other: "Node") -> "Node":
        other = other if isinstance(other, Node) else Node(other)
        result = Node(self.value + other.value, (self, other), "+")

        def __grad():
            self.gradient = self.gradient + result.gradient

            g_other = result.gradient
            # gradient of the bias is the sum of the gradients of the
            # bias for each row of the batch.
            # if broadcast happens, sum over the broadcasted axes.
            if other.value.shape != result.value.shape:
                # Common case: (N, M) + (M,)
                g_other = np.sum(g_other, axis=0)

            other.gradient = other.gradient + g_other

        result.__grad = __grad
        return result

    def relu(self) -> "Node":
        result = Node(np.maximum(self.value, 0), (self,), "ReLU")

        def __grad():
            self.gradient = self.gradient + result.gradient * (self.value > 0)

        result.__grad = __grad
        return result

    def softmax(self) -> "Node":
        # stable softmax over last axis
        x = self.value

        # trick to prevent overflow of exp is to subtract
        # X_max from inputs before running softmax
        x = x - np.max(x, axis=-1, keepdims=True)

        expx = np.exp(x)
        s = expx / np.sum(expx, axis=-1, keepdims=True)

        result = Node(s, (self,), "SoftMax")

        def __grad():
            # JVP: dL/dx = s * (g - sum(g*s))
            g = result.gradient
            dot = np.sum(g * s, axis=-1, keepdims=True)
            self.gradient = self.gradient + s * (g - dot)

        result.__grad = __grad
        return result

    def reset_grad(self) -> None:
        if not self.nodes:
            raise RuntimeError("Not the root node!!")

        for node in self.nodes:
            node.gradient = node.gradient * 0

    def propagate_back(self, gradient) -> None:
        if not self.nodes:
            raise RuntimeError("Not the root node!!")

        self.gradient = gradient
        for node in reversed(self.nodes):
            node.__grad()

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"\nNode {id(self)} {self.node_type} \n \
                value= \n {self.value} \n grad= \n {self.gradient}\n"


def topo_sort(root: Node) -> list[Node]:
    result = []

    def dfs(node: Node):
        for child in node.children:
            if not child.visited:
                dfs(child)
        node.visited = True
        result.append(node)

    dfs(root)

    for node in result:
        node.visited = False

    root.nodes = result
    return result
