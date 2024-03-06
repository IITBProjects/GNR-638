import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset

class CUB(VisionDataset):
    def __init__(self, df: pd.DataFrame, root = None, transforms = None, transform = None, target_transform = None):
        super().__init__(root, transforms, transform, target_transform)
        self.df = df
        self.df.reset_index(inplace = True,drop = True)

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
            return image, target - 1
        else:
            raise ValueError(f'{index} >=  size of dataframe')

    def get_image(self, image_id):
        row = self.df[self.df['image_id'] == image_id]
        if not row.empty:
            image_path = row.iloc[0]['image_path']
            image_pil = Image.open(image_path).convert('RGB')
            target = row.iloc[0]['class_id']
            image = self.transform(image_pil)
            if self.target_transform:
                target = self.target_transform(target)
            return image, target - 1
        else:
            return None,None
