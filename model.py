"""
model.py

4-layer CNN for 7-class DermaMNIST classification on 128x128
RGB dermatoscopic images.
"""

import torch
import torch.nn as nn


class DiseaseTypeCNN(nn.Module):
    """
    4-layer CNN for 7-class disease type classification on 128x128
    RGB microscopy images (DermaMNIST)
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        # Block 1:
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2:
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3:
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Block 4:
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Shared pooling layer - the same operation reused after each conv
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout before the classifier head
        self.dropout = nn.Dropout(p=0.3)

        # Flattened dimension: 128x128 input -> four 2x2 pools -> 8x8 spatial, 256 channels
        self.fc1 = nn.Linear(in_features=256 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)


    def forward(self, x):
        # Block 1:
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Block 2:
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Block 3:
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Block 4:
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Flatten 
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DiseaseTypeCNN(num_classes=7)
    dummy_input = torch.randn(4, 3, 128, 128)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

