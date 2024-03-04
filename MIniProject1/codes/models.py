import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class CustomModel(nn.Module):
    def __init__(self, pretrained_model_name, num_classes, freeze_layers=[]):
        super(CustomModel, self).__init__()

        self.pretrained_model_name = pretrained_model_name
        self.num_classes = num_classes
        # Load the pretrained model
        self.pretrained_model = self.load_pretrained_model()

        # Modify layers based on parameters
        self.modify_final_classification_layer()
        self.freeze_layers(freeze_layers)

    def load_pretrained_model(self):
        if self.pretrained_model_name == 'mobilenetv2':
            pretrained_model = models.mobilenet_v2(pretrained=True)
        elif self.pretrained_model_name == 'efficientnetb0':
            pretrained_model = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            raise ValueError("Invalid pretrained model name")

        return pretrained_model

    def modify_final_classification_layer(self):
        if self.pretrained_model_name =='mobilenetv2':
            num_features = self.pretrained_model.classifier[1].in_features
            self.pretrained_model.classifier = nn.Linear(num_features, self.num_classes)
        elif self.pretrained_model_name == 'efficientnetb0':
            num_features = self.pretrained_model._fc.in_features
            self.pretrained_model._fc = nn.Linear(num_features, self.num_classes)

    def freeze_layers(self, freeze_layers):
        # Freeze specified layers
        if freeze_layers:
            for name, param in self.pretrained_model.named_parameters():
                if any(freeze_layer in name for freeze_layer in freeze_layers):
                    param.requires_grad = False

    def forward(self, x):
        return self.pretrained_model(x)
