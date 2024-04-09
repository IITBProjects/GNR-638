import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch

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

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:depth-7])
        # Freeze encoder layers
        for param in self.encoder.parameters():
            param.requires_grad = False
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
