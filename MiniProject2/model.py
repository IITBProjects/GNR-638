import torch.nn as nn
import torchvision.models as models


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


class EncoderDecoder(nn.Module):
    def __init__(self, kernel_size = 5, filters = 32):
        super(EncoderDecoder, self).__init__()

        encoder_layers, decoder_layers = [], []
        for i in range(len(filters)):
            encoder_layers += [nn.Conv2d(3 if i == 0 else filters[i-1], filters[i], kernel_size = kernel_size), nn.ReLU()]
            decoder_layers += [nn.ReLU(), nn.ConvTranspose2d(filters[i], 3 if i == 0 else filters[i-1], kernel_size = kernel_size)]
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers[::-1])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DeblurResnet(nn.Module):
    def __init__(self, depth = 5):
        super(DeblurResnet, self).__init__()

        resnet = models.resnet18(pretrained=True)
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
