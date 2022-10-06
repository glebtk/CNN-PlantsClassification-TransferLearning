import torch
import config
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="zeros"):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=True,
                      padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super().__init__()

        self.dense = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True) if activation else nn.Identity(),
            # nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.dense(x)


class Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=50, conv_features=None, dense_features=None):
        super().__init__()

        if conv_features is None:
            conv_features = [64, 128, 256]

        if dense_features is None:
            dense_features = [256, 256, 128, 64]

        conv_layers = []

        for feature in conv_features:
            conv_layers.append(ConvBlock(in_channels, feature, stride=1))
            conv_layers.append(ConvBlock(feature, feature, stride=1))
            conv_layers.append(ConvBlock(feature, feature, stride=1))
            conv_layers.append(ConvBlock(feature, feature, stride=1 if feature == conv_features[-1] else 2))

            in_channels = feature

        conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1))

        self.conv = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()

        dense_layers = []

        in_channels = dense_features[0]
        for feature in dense_features[1:]:
            dense_layers.append(DenseBlock(in_channels, feature))
            in_channels = feature

        dense_layers.append(DenseBlock(in_channels, out_channels, activation=False))

        self.dense = nn.Sequential(*dense_layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        # return torch.sigmoid(x)
        return x


def test():
    model = Model()

    # print(model)
    # summary(model, depth=5)
    images = torch.rand([2, config.IN_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE], dtype=torch.float32)

    after_nn = model(images)

    print(torch.softmax(after_nn[0], dim=0))
    # print(after_nn[0].shape)
    # print(after_nn[0])
    print("ok")


if __name__ == "__main__":
    test()
