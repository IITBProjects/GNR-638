import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from model import VariationalAutoencoder
from utils import Utils

class VAEPipeline:
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
        train_config = self.config['vae_train']
        self.model = VariationalAutoencoder(**train_config['vae_model']).to(self.device)
        print('Number of parameters: %d' % sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=train_config['lr'], weight_decay=1e-5)

        train_dataloader = DataLoader(self.train_dataset, batch_size=train_config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=train_config['batch_size'], shuffle=True)

        train_losses, test_losses = [], []

        for epoch in range(train_config['epochs']):
            train_loss = self.run_epoch(train_dataloader, True)
            test_loss = self.run_epoch(test_dataloader, False)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print(f"Epoch [{epoch+1}/{train_config['epochs']}]: train_loss: {train_loss}, test_loss: {test_loss}")

        Utils.plot(train_losses, test_losses, 'VAE Loss', './plots/vae_loss.png')
        torch.save(self.model.state_dict(), './models/vae_model.pth')

    def run_epoch(self, dataloader: DataLoader, train: bool):
        self.model.train() if train else self.model.eval()
        total_loss = 0

        with torch.set_grad_enabled(train):
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(self.device)
                image_batch_recon, latent_mu, latent_logvar = self.model(image_batch)
                loss = Utils.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar, self.config['vae_train']['variational_beta'])

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

        return total_loss / len(dataloader.dataset)

    def visualise(self):
        train_config = self.config['vae_train']
        self.model = VariationalAutoencoder(**train_config['vae_model'])
        self.model.load_state_dict(torch.load('./models/vae_model.pth'))
        test_dataloader = DataLoader(self.test_dataset, batch_size=train_config['batch_size'], shuffle=True)

        images, _ = next(iter(test_dataloader))
        outputs, _, _ = self.model(images)
        Utils.show_images(images, 5, 10, './plots/orignal_images.png')
        Utils.show_images(outputs, 5, 10, './plots/reconstructed_images.png')

