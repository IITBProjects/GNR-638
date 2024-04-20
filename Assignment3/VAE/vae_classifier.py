import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from model import VariationalAutoencoder, VAEClassificationModel
from utils import Utils

class VAEClassifier:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.random.manual_seed(0)

    def create_dataset(self):
        img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.train_dataset = MNIST(root=self.config['dataset']['mnist_path'], download=True, train=True, transform=img_transform)
        self.test_dataset = MNIST(root=self.config['dataset']['mnist_path'], download=True, train=False, transform=img_transform)

    def train(self):
        vae_model = VariationalAutoencoder(**self.config['vae_train']['vae_model']).to(self.device)
        vae_model.load_state_dict(torch.load('./models/vae_model.pth'))

        train_config = self.config['classifier_train']
        self.model = VAEClassificationModel(vae_model).to(self.device)
        print('Number of parameters: %d' % sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=train_config['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()

        train_dataloader = DataLoader(self.train_dataset, batch_size=train_config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=train_config['batch_size'], shuffle=True)

        train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

        for epoch in range(train_config['epochs']):
            train_loss, train_accuracy = self.run_epoch(train_dataloader, True)
            test_loss, test_accuracy = self.run_epoch(test_dataloader, False)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            print(f"Epoch [{epoch+1}/{train_config['epochs']}]: train_loss: {train_loss}, test_loss: {test_loss}, train_accuracy: {train_accuracy}, test_accuracy: {test_accuracy}")

        Utils.plot(train_losses, test_losses, "Classifier Loss", './plots/classifier_loss.png')
        Utils.plot(train_accuracies, test_accuracies, "Classifier Accuracy", './plots/classifier_accuracy.png')

        torch.save(self.model.state_dict(), './models/classifier.pth')

    def run_epoch(self, dataloader: DataLoader, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss, total_accuracy = 0, 0

        with torch.set_grad_enabled(train):
            for image_batch, labels in dataloader:
                image_batch, labels = image_batch.to(self.device), labels.to(self.device)
                outputs = self.model(image_batch)
                _, predictions = torch.max(outputs.data, 1)
                loss = self.criterion(outputs, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                total_accuracy += (predictions == labels).sum().item()

        return total_loss / len(dataloader.dataset), total_accuracy / len(dataloader.dataset)
