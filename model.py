import config
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights
from torchvision.models import DenseNet121_Weights
from torchvision.models import DenseNet201_Weights
from torchvision.models import ConvNeXt_Tiny_Weights
from torchvision.models import ResNet18_Weights
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models import MobileNet_V3_Small_Weights


class Model(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        if model_name == "alexnet":
            alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

            # "Замораживаем" градиенты в feature extractor сети:
            for param in alexnet.features.parameters():
                param.requires_grad = False

            # Заменяем последний слой:
            in_features = alexnet.classifier[-1].in_features
            alexnet.classifier[-1] = nn.Linear(in_features=in_features, out_features=config.OUT_FEATURES)

            self.model = alexnet

        if model_name == "convnext_tiny":
            convnext = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

            # "Замораживаем" градиенты в feature extractor сети:
            for param in convnext.features.parameters():
                param.requires_grad = False

            # Заменяем последний слой:
            in_features = convnext.classifier[-1].in_features
            convnext.classifier[-1] = nn.Linear(in_features=in_features, out_features=config.OUT_FEATURES)

            self.model = convnext

        if model_name == "densenet121":
            densenet121 = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

            # "Замораживаем" все слои нейросети:
            for param in densenet121.features.parameters():
                param.requires_grad = False

            # Заменяем классификатор:
            in_features = densenet121.classifier.in_features
            densenet121.classifier = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=int(in_features / 2)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=int(in_features / 2), out_features=int(in_features / 4)),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(in_features=int(in_features / 4), out_features=config.OUT_FEATURES)
            )

            self.model = densenet121

        if model_name == "densenet201":
            densenet201 = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)

            # "Замораживаем" все слои нейросети:
            for param in densenet201.features.parameters():
                param.requires_grad = False

            # Заменяем классификатор:
            in_features = densenet201.classifier.in_features
            densenet201.classifier = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=int(in_features / 2)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=int(in_features / 2), out_features=int(in_features / 4)),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(in_features=int(in_features / 4), out_features=config.OUT_FEATURES)
            )

            self.model = densenet201

        if model_name == "resnet18":
            resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)

            # "Замораживаем" все слои нейросети:
            for param in resnet18.parameters():
                param.requires_grad = False

            # Заменяем последний слой:
            in_features = resnet18.fc.in_features
            resnet18.fc = nn.Linear(in_features=in_features, out_features=config.OUT_FEATURES)

            self.model = resnet18

        if model_name == "mobilenet_v3_small":
            mobilenet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

            # "Замораживаем" градиенты в feature extractor сети:
            for param in mobilenet.features.parameters():
                param.requires_grad = False

            # Заменяем последний слой:
            in_features = mobilenet.classifier[-1].in_features
            mobilenet.classifier[-1] = nn.Linear(in_features=in_features, out_features=config.OUT_FEATURES)

            self.model = mobilenet

        if model_name == "mobilenet_v3_large":
            mobilenet = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

            # "Замораживаем" градиенты в feature extractor сети:
            for param in mobilenet.features.parameters():
                param.requires_grad = False

            # Заменяем последний слой:
            in_features = mobilenet.classifier[-1].in_features
            mobilenet.classifier[-1] = nn.Linear(in_features=in_features, out_features=config.OUT_FEATURES)

            self.model = mobilenet

    def forward(self, x):
        return self.model(x)


def test():
    model = Model(model_name="mobilenet_v3_large")


if __name__ == "__main__":
    test()
