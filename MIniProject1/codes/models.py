import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes,freeze_pretrianed=True,freeze_layers=[],dropout_prob=0.35):
        super(EfficientNetModel, self).__init__()

        # Load the pretrained model
        pretrained_model = EfficientNet.from_pretrained('efficientnet-b2')
        # Freeze all original parameters 
        if freeze_pretrianed:
            for param in pretrained_model.parameters():
                param.requires_grad = False
        self.pretrained_model = pretrained_model

        # Modify the classifcation layers
        num_features = self.pretrained_model._fc.in_features
        self.pretrained_model._fc = nn.Sequential(
            nn.Linear(num_features, 512),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )
        
        # Freeze specified layers
        if freeze_layers:
            for name, param in self.pretrained_model.named_parameters():
                if any(freeze_layer in name for freeze_layer in freeze_layers):
                    param.requires_grad = False

    def forward(self, x):
        return self.pretrained_model(x)
    

class MobileNetModel(nn.Module):
    def __init__(self, num_classes,freeze_pretrianed=True,freeze_layers=[], dropout_prob=0.35):
        super(MobileNetModel, self).__init__()

        # Load the pretrained model
        pretrained_model = models.mobilenet_v3_large(pretrained=True)
        # Freeze all original parameters 
        if freeze_pretrianed:
            for param in pretrained_model.parameters():
                param.requires_grad = False
        self.pretrained_model = pretrained_model

        # Modify the classifcation layers
        num_features = self.pretrained_model.classifier[0].in_features
        self.pretrained_model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes),
        )

        # Freeze specified layers
        if freeze_layers:
            for param in pretrained_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.pretrained_model(x)
    

class InceptionNetModel(nn.Module):
    def __init__(self, num_classes, freeze_pretrained=True, freeze_layers=[], dropout_prob=0.35):
        super(InceptionNetModel, self).__init__()

        # Load the pretrained model
        pretrained_model = models.googlenet(pretrained=True)
        
        # Freeze all original parameters 
        if freeze_pretrained:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        self.pretrained_model = pretrained_model

        # Modify the classification layers
        num_features = pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )

        # Freeze specified layers
        if freeze_layers:
            for name, param in self.pretrained_model.named_parameters():
                if any(freeze_layer in name for freeze_layer in freeze_layers):
                    param.requires_grad = False

    def forward(self, x):
        return self.pretrained_model(x)


class ResNetModel(nn.Module):
    def __init__(self, num_classes, freeze_pretrained=True, freeze_layers=[], dropout_prob=0.35):
        super(ResNetModel, self).__init__()

        # Load the pretrained model
        pretrained_model = models.resnet18(pretrained=True)
        
        # Freeze all original parameters 
        if freeze_pretrained:
            for param in pretrained_model.parameters():
                param.requires_grad = False

        self.pretrained_model = pretrained_model

        # Modify the classification layers
        num_features = pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),  
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )

        # Freeze specified layers
        if freeze_layers:
            for name, param in self.pretrained_model.named_parameters():
                if any(freeze_layer in name for freeze_layer in freeze_layers):
                    param.requires_grad = False

    def forward(self, x):
        return self.pretrained_model(x)