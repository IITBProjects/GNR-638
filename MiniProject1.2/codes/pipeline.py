from codes import Utils, CUB, models
import os
from datetime import datetime
import pandas as pd
import torch
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class Pipeline:
    def __init__(self, config):
        self.config = config
        dataset_path = config['dataset_path']

        labels = Utils.read_file(f'{dataset_path}/image_class_labels.txt', ['image_id', 'class_id'])
        train_test = Utils.read_file(f'{dataset_path}/train_test_split.txt', ['image_id', 'is_training_image'])
        images = Utils.read_file(f'{dataset_path}/images.txt', ['image_id', 'image_name'])
        classes = Utils.read_file(f'{dataset_path}/classes.txt', ['class_id', 'class_name'])

        self.data = pd.merge(pd.merge(images, train_test, on='image_id'), pd.merge(labels, classes, on='class_id'), on='image_id')
        self.data['image_path'] = self.data['image_name'].apply(lambda x: os.path.join(dataset_path, 'images', x))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_dataloder(self):
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.config['transform']['height']),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
            transforms.RandomRotation(degrees = 10),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.config['transform']['normalize_mean'], std = self.config['transform']['normalize_std']),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.config['transform']['height'], self.config['transform']['width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.config['transform']['normalize_mean'], std = self.config['transform']['normalize_std']),
        ])

        train_dataset = CUB(df = self.data[self.data['is_training_image'] == 1], transform = train_transform)
        test_dataset = CUB(df = self.data[self.data['is_training_image'] == 0], transform = test_transform)

        self.train_loader = DataLoader(train_dataset, batch_size = self.config['train']['batch_size'], shuffle = True, num_workers = 4)
        self.test_loader = DataLoader(test_dataset, batch_size = self.config['train']['batch_size'], num_workers = 4)
        self.num_classes = len(self.data['class_id'].unique())

        print("Train Dataset length:", train_dataset.__len__())
        print("Test Dataset length:", test_dataset.__len__())
        print("Dataset  Classes:", self.num_classes)

    def init_train(self):
        self.model: torch.nn.Module = getattr(models, self.config['train']['model'])(
            num_classes = self.num_classes,
            freeze_layers = self.config['train']['freeze_layers']
        ).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.0005)
        self.train_epochs = 0

        if self.config['train']['checkpoint_path']:
            checkpoint_path = os.path.join(self.config['output_dir'], self.config['train']['model'], self.config['train']['checkpoint_path'])
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_epochs = checkpoint['epochs']
            print("Loaded model from checkpoint:", checkpoint_path)

        print("Model Summary:")
        summary(self.model, (3, self.config['transform']['height'], self.config['transform']['width']))

    def save(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epochs': self.train_epochs
        }, os.path.join(self.config['output_dir'], self.config['train']['model'], self.config['train']['checkpoint_save']))

    def train(self):
        train_loss_list, test_loss_list = [], []
        train_accuracy_list, test_accuracy_list = [], []
        num_epochs = self.config['train']['epochs']

        start = datetime.now()
        for epoch in range(num_epochs):
            _, _ = self.run_epoch(self.train_loader, True)

            train_loss, train_accuracy = self.run_epoch(self.train_loader, False)
            test_loss, test_accuracy = self.run_epoch(self.test_loader, False)

            print(
                f"Epoch [{epoch + 1}/{num_epochs}]",
                f"Mean Train Loss: {train_loss:.4f}, Mean Test Loss: {test_loss:.4f}",
                f"Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%",
                sep = ', '
            )

            train_accuracy_list.append(train_accuracy)
            test_accuracy_list.append(test_accuracy)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
        end = datetime.now()
        print(f"Time elapsed: {(end - start).total_seconds()}")

        return train_loss_list, test_loss_list, train_accuracy_list, test_accuracy_list
    
    def test(self):
        self.model.eval()
        test_loss, test_accuracy = self.run_epoch(self.test_loader, False, log = 'Testing Eval')

        print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
    
    def run_epoch(self, dataloader: DataLoader, train):
        self.model.train() if train else self.model.eval()
        running_loss = 0.
        correct_predictions, total_predictions = 0, 0

        with torch.set_grad_enabled(train):
            for batch, (inputs, labels) in enumerate(dataloader):
                batch_size = labels.size(0)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if train: self.optimizer.zero_grad()
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                correct = (predicted == labels).sum().item()
                loss = self.criterion(outputs, labels)
                
                correct_predictions += correct
                total_predictions += batch_size
                running_loss += batch_size * loss.item()

                if train:
                    loss.backward()
                    self.optimizer.step()

                    if batch % self.config['train']['batch_logs'] == 0:
                        print(
                            f"Training Batch [{batch}/{len(dataloader)}]",
                            f"Batch Size: {batch_size}",
                            f"Batch Mean Loss: {loss.item():.5f}",
                            f"Batch Accuracy: {(correct / batch_size)*100:.2f}%",
                            f"Rolling Epoch Accuracy: {(correct_predictions / total_predictions) * 100:.2f}%",
                            sep = '    '
                        )

        return running_loss / total_predictions, correct_predictions / total_predictions
