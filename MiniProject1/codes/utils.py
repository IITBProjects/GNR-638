import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt


class Utils:
    def read_file(path, columns):
        df = pd.read_csv(path, header=None, sep=" ")
        df.columns = columns
        return df
    
    def freeze(model: nn.Module = None, freeze_layers = []):
        for name, param in model.named_parameters():
            if any(freeze_layer in name for freeze_layer in freeze_layers):
                param.requires_grad = False

    def freeze_all(model: nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def plot_metrics(train_data, test_data, name, title, path):
        epochs = range(1, len(train_data) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_data, label = f'Train {name}')
        plt.plot(epochs, test_data, label = f'Test {name}')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        plt.savefig(path)
        plt.close()
