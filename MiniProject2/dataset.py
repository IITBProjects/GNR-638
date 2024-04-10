import os
import random
from skimage.io import imread
from torch.utils.data import Dataset
from utils import Utils


class StaticImageDataset(Dataset):
    def __init__(self, setA_dir, setB_dir, image_paths):
        super().__init__()
        self.setA_dir = setA_dir
        self.setB_dir = setB_dir
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        setA_image = imread(os.path.join(self.setA_dir, *self.image_paths[index]))
        setB_image = imread(os.path.join(self.setB_dir, *self.image_paths[index]))
        return Utils.image_to_tensor(setB_image), Utils.image_to_tensor(setA_image)


class RuntimeImageDataset(Dataset):
    def __init__(self, dir, image_paths, input_transform = None, output_transform = None):
        self.dir = dir
        self.image_paths = image_paths
        self.input_transform = input_transform
        self.output_transform = output_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = imread(os.path.join(self.dir, *self.image_paths[index])) / 255
        input_image = self.input_transform(image) if self.input_transform else image
        output_image = self.output_transform(image) if self.output_transform else image
        return Utils.image_to_tensor(input_image), Utils.image_to_tensor(output_image)


class VideoDataset(Dataset):
    def __init__(self, dir, image_dirs, input_transform = None, output_transform = None):
        self.dir = dir
        self.image_dirs = image_dirs
        self.input_transform = input_transform
        self.output_transform = output_transform

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, index):
        image_path = random.choice(os.listdir(os.path.join(self.dir, self.image_dirs[index])))
        image = imread(os.path.join(self.dir, self.image_dirs[index], image_path)) / 255
        input_image = self.input_transform(image) if self.input_transform else image
        output_image = self.output_transform(image) if self.output_transform else image
        return Utils.image_to_tensor(input_image), Utils.image_to_tensor(output_image)
