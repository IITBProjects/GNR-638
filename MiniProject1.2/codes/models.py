from .utils import Utils
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class EfficientNetModel(nn.Module):
    def __init__(self, num_classes, freeze_pretrianed = True, freeze_layers = [], dropout_prob=0.35):
        super(EfficientNetModel, self).__init__()

        # Load the pretrained model
        self.pretrained_model = EfficientNet.from_pretrained('efficientnet-b2')
        if freeze_pretrianed:
            Utils.freeze_all(self.pretrained_model)

        # Modify the classifcation layers
        num_features = self.pretrained_model._fc.in_features
        self.pretrained_model._fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, num_classes)
        )
        Utils.freeze(self.pretrained_model, freeze_layers)

    def forward(self, x):
        return self.pretrained_model(x)
    

class MobileNetModel(nn.Module):
    def __init__(self, num_classes, freeze_pretrianed = True, freeze_layers = [], dropout_prob = 0.35):
        super(MobileNetModel, self).__init__()

        # Load the pretrained model
        self.pretrained_model = models.mobilenet_v3_large(pretrained = True)
        if freeze_pretrianed:
            Utils.freeze_all(self.pretrained_model)

        # Modify the classifcation layers
        num_features = self.pretrained_model.classifier[0].in_features
        self.pretrained_model.classifier = nn.Sequential(
                nn.Linear(num_features, 512),  
                nn.Hardswish(inplace = True),
                nn.Dropout(p = dropout_prob),
                nn.Linear(512, num_classes)
        )
        Utils.freeze(self.pretrained_model, freeze_layers)

    def forward(self, x):
        return self.pretrained_model(x)
    

class InceptionNetModel(nn.Module):
    def __init__(self, num_classes, freeze_pretrained = True, freeze_layers = [], dropout_prob = 0.35):
        super(InceptionNetModel, self).__init__()

        # Load the pretrained model
        self.pretrained_model = models.googlenet(pretrained = True)
        if freeze_pretrained:
            Utils.freeze_all(self.pretrained_model)

        # Modify the classification layers
        num_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),  
            nn.ReLU(inplace = True),
            nn.Dropout(p = dropout_prob),
            nn.Linear(512, num_classes)
        )
        Utils.freeze(self.pretrained_model, freeze_layers)

    def forward(self, x):
        return self.pretrained_model(x)


class DenseNetModel(nn.Module):
    def __init__(self, num_classes, freeze_pretrained = True, freeze_layers = [], dropout_prob = 0.35):
        super(DenseNetModel, self).__init__()

        # Load the pretrained model
        self.pretrained_model = models.densenet121(pretrained = True)
        if freeze_pretrained:
            Utils.freeze_all(self.pretrained_model)

        # Modify the classification layers
        num_features = self.pretrained_model.classifier.in_features
        self.pretrained_model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),  
            nn.ReLU(inplace = True),
            nn.Dropout(p = dropout_prob),
            nn.Linear(512, num_classes)
        )
        Utils.freeze(self.pretrained_model, freeze_layers)

    def forward(self, x):
        return self.pretrained_model(x)
