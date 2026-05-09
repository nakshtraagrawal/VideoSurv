"""
Conv Autoencoder — baseline anomaly detector.
Train on normal frames only. High reconstruction error = anomaly.
"""
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, in_ch=1, latent=512):
        super().__init__()

        def conv_block(ic, oc, stride=2):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride, 1),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2),
            )

        def deconv_block(ic, oc):
            return nn.Sequential(
                nn.ConvTranspose2d(ic, oc, 4, 2, 1),
                nn.BatchNorm2d(oc),
                nn.ReLU(),
            )

        self.enc = nn.Sequential(
            conv_block(in_ch, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.fc_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, latent),
        )
        self.fc_dec = nn.Sequential(
            nn.Linear(latent, 256 * 16 * 16),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            deconv_block(256, 128),
            deconv_block(128, 64),
            deconv_block(64, 32),
            nn.ConvTranspose2d(32, in_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.fc_enc(self.enc(x))
        x_hat = self.dec(self.fc_dec(z).view(-1, 256, 16, 16))
        return x_hat

    def anomaly_score(self, x):
        with torch.no_grad():
            return torch.mean((x - self(x)) ** 2, dim=[1, 2, 3])
