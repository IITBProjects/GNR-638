import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio


class Utils:

    @staticmethod
    def create_setA(dataset_config, start_from = 0, end_at = 239):
        images_dirs = dataset_config['images_dirs']
        os.makedirs(dataset_config['setA'], exist_ok=True)

        for images_dir in sorted(os.listdir(images_dirs))[start_from:end_at+1]:
            src_dir = os.path.join(images_dirs, images_dir)
            dest_dir = os.path.join(dataset_config['setA'], images_dir)
            os.makedirs(dest_dir, exist_ok=True)

            for image_path in sorted(os.listdir(src_dir)):
                image = imread(os.path.join(src_dir, image_path))
                image = resize(image, (dataset_config['image_size'][0], dataset_config['image_size'][1]))
                image = (image * 255).astype(np.uint8)
                imsave(os.path.join(dest_dir, image_path), image)

    @staticmethod
    def create_setB(dataset_config, start_from = 0, end_at = 239):
        setA_dir = dataset_config['setA']
        os.makedirs(dataset_config['setB'], exist_ok=True)

        for images_dir in sorted(os.listdir(setA_dir))[start_from:end_at+1]:
            src_dir = os.path.join(setA_dir, images_dir)
            dest_dir = os.path.join(dataset_config['setB'], images_dir)
            os.makedirs(dest_dir, exist_ok=True)

            for image_path in sorted(os.listdir(src_dir)):
                image = imread(os.path.join(src_dir, image_path))
                image = Utils.apply_gaussian(image, dataset_config['gaussian_filters'])
                image = (image * 255).astype(np.uint8)
                imsave(os.path.join(dest_dir, image_path), image)

    @staticmethod
    def apply_gaussian(image, filters):
        kernel = random.choice(filters)
        return gaussian(image, sigma = kernel['sigma'], truncate = kernel['kernel_size'], channel_axis = -1)
    
    @staticmethod
    def image_to_tensor(image):
        if image.max() > 1: image = image / image.max()
        return torch.tensor(image, dtype = torch.float32).permute(2, 0, 1)
    
    @staticmethod
    def tensor_to_image(tensor):
        image = tensor.permute(1, 2, 0).detach().numpy()
        if image.max() <= 1: image *= 255
        return np.uint8(image)
    
    @staticmethod
    def train_test_split(image_dirs, test_split, num_dirs = 240):
        train_image_paths, test_image_paths = [], []
        for image_dir in sorted(os.listdir(image_dirs))[:num_dirs]:
            image_paths = [(image_dir, image_path) for image_path in os.listdir(os.path.join(image_dirs, image_dir))]
            random.shuffle(image_paths)
            train_size = int((1 - test_split) * len(image_paths))
            
            train_image_paths += image_paths[:train_size]
            test_image_paths += image_paths[train_size:]
        return train_image_paths, test_image_paths
    
    @staticmethod
    def psnr_tensor(y1, y2):
        return peak_signal_noise_ratio(Utils.tensor_to_image(y1), Utils.tensor_to_image(y2))
    
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
