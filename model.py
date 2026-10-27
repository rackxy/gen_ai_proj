"""
model.py — Backend: Model definitions (Generator + Discriminator)
DCGAN for Fashion-MNIST image generation
"""

import torch
import torch.nn as nn

LATENT_DIM = 100
CHANNELS   = 1
FEATURES_G = 64
FEATURES_D = 64


class Generator(nn.Module):
    """
    Generative model: noise vector → fashion image (28x28 grayscale)
    """
    def __init__(self, latent_dim=LATENT_DIM, features=FEATURES_G):
        super().__init__()
        self.net = nn.Sequential(
            self._block(latent_dim, features * 4, 4, 1, 0),
            self._block(features * 4, features * 2, 3, 2, 1),
            self._block(features * 2, features,     4, 2, 1),
            nn.ConvTranspose2d(features, CHANNELS, 4, 2, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _block(in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    Discriminative model: image → real/fake score
    """
    def __init__(self, features=FEATURES_D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            self._block(features,     features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 3, 2, 1),
            nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _block(in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    if "Conv" in m.__class__.__name__:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in m.__class__.__name__:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
