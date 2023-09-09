import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MovePredictModel(nn.Module):
    def __init__(self, num_classes, backbone="resnet18", pretrained=True):
        super.__init__()
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        elif backbone == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise Exception("Only ResNet series available.")

        self.fc_layer = nn.Sequential(
            nn.Linear(1000, 512), 
            nn.Linear(512, 128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc_layer(x)
        return x

