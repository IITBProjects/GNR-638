import os
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from skimage.io import imread, imsave
# from torchvision.models import SSIM

from dataset import RuntimeImageDataset, VideoDataset
from utils import Utils
from model import DeblurModel, EncoderDecoder, DeblurResnet ,UNet, GAN, CNNModel, SRCNN
from mimo_unet import MIMOUNet, MIMOUNetPlus
from resnet import ResNet16


MODELS = {
    'srcnn': SRCNN,
    'deblur_model': DeblurModel,
    'encoder_decoder': EncoderDecoder,
    'resnet': DeblurResnet,
    'mimo_unet': MIMOUNet,
    'mimo_unet_plus': MIMOUNetPlus,
    'unet': UNet,
    'gan':GAN,
    'cnn':CNNModel,
    'resnet16': ResNet16
}

LOSS_FUNCTIONS = {
    'l1': nn.L1Loss(),
    'mse': nn.MSELoss(),
    'mimo': lambda x, y: Utils.get_mimo_loss(nn.L1Loss(), x, y),
    'logcosh': lambda x, y: torch.mean(torch.log(torch.cosh(x - y)))
}


class DeblurImages:
    def __init__(self, config):
        self.config = config
        random.seed(0)
        torch.random.manual_seed(0)

    def create_datasets(self):
        dataset_config = self.config['dataset']        
        train_image_paths, test_image_paths = Utils.train_test_split(dataset_config['setA'], dataset_config['test_split'], dataset_config['num_dirs'])

        input_transform = lambda image: Utils.apply_gaussian(image, dataset_config['gaussian_filters'])
        self.train_dataset = RuntimeImageDataset(dataset_config['setA'], train_image_paths, input_transform = input_transform)
        self.test_dataset = RuntimeImageDataset(dataset_config['setA'], test_image_paths, input_transform = input_transform)
        psnr = lambda x: sum([Utils.psnr_tensor(x.__getitem__(i)[0], x.__getitem__(i)[1]) for i in range(x.__len__())]) / x.__len__()

        print(f"Train size: {len(self.train_dataset)}, Test size: {len(self.test_dataset)}")
        print(f"PSNR train: {psnr(self.train_dataset)}, PSNR test: {psnr(self.test_dataset)}")

    def create_datasets_new(self):
        dataset_config = self.config['dataset']        
        train_image_paths, test_image_paths = Utils.train_test_split_dir(dataset_config['setA'], dataset_config['test_split'], dataset_config['num_dirs'])

        input_transform = lambda image: Utils.apply_gaussian(image, dataset_config['gaussian_filters'])
        self.train_dataset = VideoDataset(dataset_config['setA'], train_image_paths, input_transform = input_transform)
        self.test_dataset = VideoDataset(dataset_config['setA'], test_image_paths, input_transform = input_transform)
        psnr = lambda x: sum([Utils.psnr_tensor(x.__getitem__(i)[0], x.__getitem__(i)[1]) for i in range(x.__len__())]) / x.__len__()

        print(f"Train size: {len(self.train_dataset)}, Test size: {len(self.test_dataset)}")
        print(f"PSNR train: {psnr(self.train_dataset)}, PSNR test: {psnr(self.test_dataset)}")

    def train(self):
        train_config = self.config['train']

        train_dataloader = DataLoader(self.train_dataset, batch_size = train_config['batch_size'], shuffle = True)
        test_dataloader = DataLoader(self.test_dataset, batch_size = train_config['batch_size'], shuffle = False)

        self.model: nn.Module = MODELS[train_config['model']](**train_config[train_config['model']])
        summary(self.model, (3, *self.config['dataset']['image_size']))
        self.criterion = LOSS_FUNCTIONS[train_config['loss_func']]
        self.optimizer = optim.Adam(self.model.parameters(), lr = train_config['lr'])
        epoch = 0
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.1, verbose=True)

        if train_config['load']:
            checkpoint = torch.load(train_config['model_save_path'])
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epochs']
            print(f"Model loaded", epoch)

        train_losses, test_losses, train_psnrs, test_psnrs = [], [], [], []

        for epoch in range(epoch, train_config['epochs']):
            train_loss, train_psnr = self.run_epoch(train_dataloader, True)
            test_loss, test_psnr = self.run_epoch(test_dataloader, False)
            self.scheduler.step(test_loss)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_psnrs.append(train_psnr)
            test_psnrs.append(test_psnr)

            print(f"Epoch [{epoch+1}/{train_config['epochs']}]: train_loss: {train_loss}, test_loss: {test_loss}, train_psnr: {train_psnr}, test_psnr: {test_psnr}")

            if (epoch + 1) % train_config['plot_interval'] == 0:
                print("Plotting loss and psnr")
                Utils.plot(train_losses, test_losses, 'Loss', './plots/loss.png')
                Utils.plot(train_psnrs, test_psnrs, 'PSNR', './plots/psnr.png')

            if (epoch + 1) % train_config['model_save_interval'] == 0:
                print("Saving model")
                torch.save({
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epochs': epoch + 1
                }, train_config['model_save_path'])

        x, y = self.test_dataset.__getitem__(0)
        y_pred = self.model(x.unsqueeze(0))[0]
        Utils.show_tensor_image(x)
        Utils.show_tensor_image(y)
        Utils.show_tensor_image(y_pred)


    def run_epoch(self, dataloader: DataLoader, train: bool):
        self.model.train() if train else self.model.eval()
        running_loss, psnr = 0.0, 0.0

        with torch.set_grad_enabled(train):
            for batch_num, (X, Y) in enumerate(dataloader):
                if train: self.optimizer.zero_grad()
                Y_pred = self.model(X)
                loss = self.criterion(Y_pred, Y)
                if train:
                    loss.backward()
                    self.optimizer.step()
                running_loss += loss.item()
                psnr += sum([Utils.psnr_tensor(Y_pred[i], Y[i]) for i in range(len(Y))])

                if batch_num % self.config['train']['batch_log_interval'] == 0 and train:
                    total = batch_num * dataloader.batch_size + len(Y)
                    print(f"Batch [{batch_num+1}/{len(dataloader)}]: loss: {running_loss / total}, psnr: {psnr / total}")

        return running_loss / len(dataloader.dataset), psnr / len(dataloader.dataset)
    
    def pred(self):
        train_config = self.config['train']

        self.model: nn.Module = MODELS[train_config['model']](**train_config[train_config['model']])
        self.model.load_state_dict(torch.load(train_config['model_save_path'])['model'])

        for image_path in sorted(os.listdir(os.path.join(train_config['pred_dir'], 'blur'))):
            image = Utils.image_to_tensor(imread(os.path.join(train_config['pred_dir'], 'blur', image_path)) / 255)
            deblur = self.model(image.unsqueeze(0))[0]
            imsave(os.path.join(train_config['pred_dir'], 'deblur', image_path), Utils.tensor_to_image(deblur))

