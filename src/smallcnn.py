import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, n_mels=128, num_classes=8):
        super().__init__()
        
        # On suppose des entrées de forme (B, 1, 128, T)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   # divise H et W par 2
            nn.Dropout(0.25)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        # ⚠️ On ne connaît pas la dimension temporelle T à l'avance.
        # On va donc utiliser un AdaptiveAvgPool2d pour se débarrasser de T.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # sortie (B, 128, 1, 1)
        
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)  # pour l’inférence / métriques

    def forward(self, x):
        # x: (B, 1, 128, T)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        x = self.global_pool(x)      # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)    # (B, 128)
        
        logits = self.fc(x)          # (B, num_classes)
        probs = self.softmax(logits) # Softmax output
        
        return logits, probs
