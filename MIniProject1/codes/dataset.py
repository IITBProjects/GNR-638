import os
import pandas as pd
from torchvision.datasets import VisionDataset
from PIL import Image

class CUB200(VisionDataset):
    def __init__(self, dataset_path, data_type='train', transform=None, target_transform=None):
       
        super().__init__(root=dataset_path, transform=transform, target_transform=target_transform)

        # Read train/test split file
        split_file = os.path.join(dataset_path, 'train_test_split.txt')
        split_df = pd.read_csv(split_file, sep=' ', header=None, names=['image_id', 'is_training_image'])

        # Read image class labels file
        labels_file = os.path.join(dataset_path, 'image_class_labels.txt')
        labels_df = pd.read_csv(labels_file, sep=' ', header=None, names=['image_id', 'class_id'])

        # Read class names file
        classes_file = os.path.join(dataset_path, 'classes.txt')
        classes_df = pd.read_csv(classes_file, sep=' ', header=None, names=['class_id', 'class_name'])

        # Read image names file
        images_file = os.path.join(dataset_path, 'images.txt')
        images_df = pd.read_csv(images_file, sep=' ', header=None, names=['image_id', 'image_name'])

        # Merge dataframes
        df = pd.merge(split_df, labels_df, on='image_id')
        df = pd.merge(df, images_df, on='image_id')
        df = pd.merge(df, classes_df, on='class_id')

        self.num_classes = len(df['class_id'].unique())
        # Filter based on data type
        if data_type == 'train':
            df = df[df['is_training_image'] == 1]
        elif data_type == 'test':
            df = df[df['is_training_image'] == 0]
        else:
            raise ValueError("Invalid data type. Use 'train' or 'test'.")

        df['image_path'] = df['image_name'].apply(lambda x: os.path.join(dataset_path, 'images', x))
        self.df = df
        self.df.reset_index(inplace=True,drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if index < len(self.df):
            row = self.df.iloc[index]
            image_path = row['image_path']
            image_pil = Image.open(image_path).convert('RGB')
            target = row['class_id']            
            image = self.transform(image_pil)
            if self.target_transform:
                target = self.target_transform(target)
            return image, target
        else:
            raise ValueError(f'{index} >=  size of dataframe')
        
    def get_image(self, image_id):
        row = self.df[self.df['image_id']==image_id]
        if not row.empty:
            image_path = row.iloc[0]['image_path']
            image_pil = Image.open(image_path).convert('RGB')
            target = row.iloc[0]['class_id']
            image = self.transform(image_pil)
            if self.target_transform:
                target = self.target_transform(target)
            return image, target
        else:
            return None,None