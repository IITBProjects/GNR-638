from .sigmoid import Sigmoid
from .linear import Linear
from .bce_loss import BinaryCrossEntropyLoss
import numpy as np

class TwoLayerMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.linear1 = Linear(input_size, hidden_size)
        self.sigmoid1 = Sigmoid()
        self.linear2 = Linear(hidden_size, output_size)
        self.sigmoid2 = Sigmoid()
        self.loss_function = BinaryCrossEntropyLoss()
        self.input =  None
        self.target = None
        self.loss = None
        self.grad_params = {}
        self.analytical_grad_params = {}


    def forward(self, inputs,target):
        self.input = inputs
        self.target = target
        linear1_output = self.linear1.forward(self.input)
        sigmoid1_output = self.sigmoid1.forward(linear1_output)
        linear2_output = self.linear2.forward(sigmoid1_output)
        sigmoid2_output = self.sigmoid2.forward(linear2_output)
        self.loss = self.loss_function.forward(sigmoid2_output,self.target)
        return self.loss

    def backward(self):
        grad_loss_input = self.loss_function.backward()
        grad_sigmoid2_input = self.sigmoid2.backward(grad_loss_input)
        grad_linear2_input = self.linear2.backward(grad_sigmoid2_input)
        grad_sigmoid1_input = self.sigmoid1.backward(grad_linear2_input)
        grad_linear1_input = self.linear1.backward(grad_sigmoid1_input)

        self.grad_params['linear2_W'] = self.linear2.grad_weights
        self.grad_params['linear2_b'] = self.linear2.grad_bias
        self.grad_params['linear1_W'] = self.linear1.grad_weights
        self.grad_params['linear1_b'] = self.linear1.grad_bias

        return self.grad_params
    
    def analytical_gradients(self, inputs, target, epsilon):
        # Initialize analytical gradients
        self.analytical_grad_params = {
            'linear1_W': np.zeros_like(self.linear1.weights),
            'linear1_b': np.zeros_like(self.linear1.bias),
            'linear2_W': np.zeros_like(self.linear2.weights),
            'linear2_b': np.zeros_like(self.linear2.bias),
        }

        for param_name, param_value in self.__dict__.items():
            if isinstance(param_value, Linear):
                # Iterate over the rows and columns of the weight matrix
                for i in range(param_value.weights.shape[0]):
                    for j in range(param_value.weights.shape[1]):
                        param_value.weights[i, j] += epsilon
                        loss_plus = self.forward(inputs, target)
                        param_value.weights[i, j] -= 2 * epsilon
                        loss_minus = self.forward(inputs, target)
                        param_value.weights[i, j] += epsilon  # Restore original value
                        self.analytical_grad_params[param_name+'_W'][i, j] = (loss_plus - loss_minus) / (2 * epsilon)

                # Iterate over the elements of the bias vector
                for j in range(param_value.bias.shape[1]):
                    param_value.bias[0, j] += epsilon
                    loss_plus = self.forward(inputs, target)
                    param_value.bias[0, j] -= 2 * epsilon
                    loss_minus = self.forward(inputs, target)
                    param_value.bias[0, j] += epsilon  # Restore original value
                    self.analytical_grad_params[param_name+'_b'][0, j] = (loss_plus - loss_minus) / (2 * epsilon)

        return self.analytical_grad_params