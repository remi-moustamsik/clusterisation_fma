import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, n_mels=128, num_classes=8):
        super().__init__()
        # Entrées: (B, 1, 128, T)
        # Bloques conv avec BatchNorm + LeakyReLU, pooling surtout sur la dimension fréquence
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2,1)),   # downsample freq mais garder le temps
            nn.Dropout(0.1)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.15)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.25)
        )

        # pool global
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, 256, 1, 1)
        self.fc = nn.Linear(256, num_classes)
        # keep softmax for inference convenience (use logits with CrossEntropyLoss)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (B, 1, 128, T)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.global_pool(x)      # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 256)

        logits = self.fc(x)          # (B, num_classes)
        probs = self._softmax(logits)
        return logits, probs
