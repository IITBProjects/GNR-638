import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(9, 9), stride=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(5, 5), stride=1, padding='same')
        
        self.relu = nn.ReLU()
        self.linear = nn.Linear(10, 3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, kernel_size = 5, filters = []):
        super(EncoderDecoder, self).__init__()

        encoder_layers, decoder_layers = [], []
        for i in range(len(filters)):
            c1, c2 = 3 if i == 0 else filters[i-1], filters[i]
            encoder_layers += [nn.Conv2d(c1, c2, kernel_size = kernel_size), nn.ReLU()]
            decoder_layers += [nn.ReLU(), nn.ConvTranspose2d(c2, c1, kernel_size = kernel_size)]
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers[::-1])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DeblurResnet(nn.Module):
    def __init__(self, depth = 5):
        super(DeblurResnet, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:depth-7])

        decoder_layers, init_filters = [], 64
        for i in range(depth):
            shape = (init_filters, 3 if i == 0 else init_filters if i == 1 else init_filters // 2)
            decoder_layers += [nn.ReLU(), nn.ConvTranspose2d(*shape, kernel_size=3, stride=2, padding=1, output_padding=1)]
            if i >= 1: init_filters *= 2
        self.decoder = nn.Sequential(*decoder_layers[::-1])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        self.encoder1 = nn.Conv2d(in_channels, 32, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        self.bottleneck = nn.Conv2d(128, 256, 3, stride=1, padding=1)

        self.decoder1 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.decoder2 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.decoder3 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1)

        self.final_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = F.relu(self.encoder1(x))
        enc2 = F.relu(self.encoder2(F.max_pool2d(enc1, kernel_size=2)))
        enc3 = F.relu(self.encoder3(F.max_pool2d(enc2, kernel_size=2)))
        bottleneck = F.relu(self.bottleneck(F.max_pool2d(enc3, kernel_size=2)))
        dec1 = F.relu(self.decoder1(bottleneck))
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec1 = F.relu(self.decoder2(dec1))
        dec2 = torch.cat([dec1, enc2], dim=1)
        dec2 = F.relu(self.decoder3(dec2))
        dec3 = torch.cat([dec2, enc1], dim=1)

        return self.final_layer(dec3)


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()

        # Generator network
        self.generator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        gen_output = self.generator(x)
        disc_output = self.discriminator(gen_output)
        return gen_output, disc_output
