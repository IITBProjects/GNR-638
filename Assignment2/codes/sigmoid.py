import numpy as np

class Sigmoid:
    def __init__(self):
        self.input = None
        self.grad_input = None
        self.output = None
        self.grad_output = None

    def forward(self, inputs):
        self.input = inputs
        self.output = 1 / (1 + np.exp(-self.input))
        return self.output

    def backward(self, grad):
        self.grad_output = grad
        self.grad_input = self.grad_output * self.output * (1 - self.output)
        return self.grad_input