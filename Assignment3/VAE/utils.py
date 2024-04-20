import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class Utils:
    @staticmethod
    def vae_loss(recon_x, x, mu, logvar, variational_beta):
        recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + variational_beta * kldivergence

    @staticmethod
    def plot(y_train, y_test, label, path):
        plt.plot(range(len(y_train)), y_train, label = 'train')
        plt.plot(range(len(y_test)), y_test, label = 'test')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(label)
        plt.title(f"{label} vs Epochs")
        plt.savefig(path)
        plt.close()

    @staticmethod
    def show_images(images, h, w, path):
        img = make_grid(images.clamp(0, 1)[:h*w], w, h).numpy()
        plt.axis('off')
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.savefig(path)
        plt.close()
