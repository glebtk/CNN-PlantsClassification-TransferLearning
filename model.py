import config
import torch
import torch.nn as nn
import torchvision
from torch.nn.modules import Identity
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.models import AlexNet_Weights
from torchvision.models import MobileNet_V3_Small_Weights


class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        if model_name == "resnet18":
            resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)

            for param in resnet18.parameters():
                param.requires_grad = False

            in_features = resnet18.fc.in_features
            resnet18.fc = nn.Linear(in_features=in_features, out_features=config.OUT_FEATURES)

            self.model = resnet18

        if model_name == "alexnet":
            alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
            # alexnet.avgpool = Identity()

            for param in alexnet.parameters():
                param.requires_grad = False

            alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=config.OUT_FEATURES)

            self.model = alexnet

        if model_name == "mobilenet_v3_small":
            mobilenet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

            for param in mobilenet.parameters():
                param.requires_grad = False

            mobilenet.classifier[3] = nn.Linear(in_features=1024, out_features=config.OUT_FEATURES)

            self.model = mobilenet

    def forward(self, x):
        return self.model(x)
