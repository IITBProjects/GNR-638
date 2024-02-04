import numpy as np

class BinaryCrossEntropyLoss:
    def __init__(self):
        self.predictions = None
        self.grad_predictions = None
        self.output = None
        self.target = None

    def forward(self, predictions, target):
        self.predictions = predictions
        self.target = target
        self.output = -np.mean(target * np.log(predictions) + (1 - target) * np.log(1 - predictions))
        return self.output

    def backward(self):
        self.grad_predictions = (-(self.target / self.predictions) + (1 - self.target) / (1 - self.predictions))/self.target.shape[0]
        return self.grad_predictions