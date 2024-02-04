import numpy as np

class Linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.grad_weights =None  
        self.bias = np.random.rand(1, output_size) 
        self.grad_bias = None 
        self.input = None
        self.grad_input = None
        self.output = None
        self.grad_output = None


    def forward(self, inputs):
        self.input = inputs
        self.output = np.matmul(self.input, self.weights) + self.bias
        return self.output

    def backward(self, grad):
        self.grad_output = grad
        self.grad_weights = np.matmul(self.input.T, self.grad_output)
        self.grad_bias = np.sum(self.grad_output, axis=0, keepdims=True)
        self.grad_input = np.matmul(self.grad_output, self.weights.T)
        return self.grad_input