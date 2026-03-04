# class GDOptimizer(Optimizer):
#     def __init__(self, lr: float = 0.01):
#         self.lr = lr

#     def step(self, params: list[Node]):
#         for p in params:
#             p.value = p.value - self.lr * p.gradient
