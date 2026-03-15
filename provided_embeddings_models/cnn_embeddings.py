"""
CNN that takes mel-spectrogram and outputs a fixed-size embedding for CatMLP.

Input: (batch, n_mels, time) — e.g. (B, 64, T)
Output: (batch, embed_dim) — e.g. (B, 128)
"""

import torch
import torch.nn as nn

from provided_embeddings_models.constants import N_MELS, EMBED_DIM


class CatEmbeddingCNN(nn.Module):
    """
    Small CNN that maps a mel-spectrogram to an embedding vector.
    Uses a few conv blocks + global pooling + linear to EMBED_DIM.
    """

    def __init__(
        self,
        in_mels: int = N_MELS,
        embed_dim: int = EMBED_DIM,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Conv blocks: (batch, in_mels, time) -> progressively smaller
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (32, m/2, t/2)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, m/4, t/4)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (128, m/8, t/8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # After 3 strides of 2: time and mel dims are reduced by 8
        # We use global avg pool so any input length works
        self.pool = nn.AdaptiveAvgPool2d(1)  # (batch, 128, 1, 1)
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (batch, n_mels, time)
        :return: (batch, embed_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)  # (batch, 1, n_mels, time)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
