import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, AvgPool2d, Dropout, Linear
import typing

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)

        self.residual_conv = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        residual = self.residual_conv(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class ResidualConvTransposeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super(ResidualConvTransposeBlock, self).__init__()
        self.conv_transpose = ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv1 = Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = BatchNorm2d(out_channels)
        self.relu = ReLU(inplace=True)

        self.residual_conv_transpose = ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=stride - 1)

    def forward(self, x):
        residual = self.residual_conv_transpose(x)

        out = self.conv_transpose(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class ResNet16(nn.Module):
    def __init__(self, input_shape: typing.Tuple[int, int, int] = (3, 256, 448)):
        super(ResNet16, self).__init__()

        # ENCODER
        self.layer1 = ResidualConvBlock(input_shape[0], 64, 3, 1)
        self.layer2 = ResidualConvBlock(64, 128, 3, 2)
        self.layer3 = ResidualConvBlock(128, 256, 3, 2)
        self.layer4 = ResidualConvBlock(256, 512, 3, 2)

        # DECODER
        self.layer5 = ResidualConvTransposeBlock(512, 256, 3, 2)
        self.layer6 = ResidualConvTransposeBlock(256, 128, 3, 2)
        self.layer7 = ResidualConvTransposeBlock(128, 64, 3, 2)
        self.layer8 = ResidualConvTransposeBlock(64, 3, 3, 1)

    def forward(self, x):
        # ENCODER
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # DECODER
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        return x

class ResNet16Dense(nn.Module):
    def __init__(self, input_shape: typing.Tuple[int, int, int]):
        super(ResNet16Dense, self).__init__()

        # ENCODER
        self.layer1 = ResidualConvBlock(input_shape[0], 64, 3, 1)
        self.layer2 = ResidualConvBlock(64, 128, 3, 2)
        self.layer3 = ResidualConvBlock(128, 256, 3, 2)

        # DENSE BOTTLENECK
        self.avg_pool = AvgPool2d(kernel_size=(8, 8))
        self.flatten = nn.Flatten()
        self.dense = Linear(256, 256)
        self.dropout = Dropout(0.2)
        self.reshape = nn.Reshape((16, 16, 1))

        # DECODER
        self.layer4 = ResidualConvTransposeBlock(256, 128, 3, 2)
        self.layer5 = ResidualConvTransposeBlock(128, 64, 3, 1)
        self.layer6 = ConvTranspose2d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ENCODER
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # DENSE BOTTLENECK
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.reshape(x)

        # DECODER
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x

class ResNet20(nn.Module):
    def __init__(self, input_shape: typing.Tuple[int, int, int]):
        super(ResNet20, self).__init__()

        # ENCODER
        self.layer1 = ResidualConvBlock(input_shape[0], 64, 3, 1)
        self.layer2 = ResidualConvBlock(64, 128, 3, 2)
        self.layer3 = ResidualConvBlock(128, 256, 3, 2)
        self.layer4 = ResidualConvBlock(256, 512, 3, 2)

        # BOTTLENECK
        self.layer5 = ResidualConvBlock(512, 1024, 3, 2)

        # DECODER
        self.layer6 = ResidualConvTransposeBlock(1024, 512, 3, 1)
        self.layer7 = ResidualConvTransposeBlock(512, 256, 3, 2)
        self.layer8 = ResidualConvTransposeBlock(256, 128, 3, 2)
        self.layer9 = ResidualConvTransposeBlock(128, 64, 3, 2)
        self.layer10 = ConvTranspose2d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ENCODER
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # BOTTLENECK
        x = self.layer5(x)

        # DECODER
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)

        return x
