import torch
import torch.nn as nn
import torch.nn.functional as F



class MicrobiomeTabularConvNet(nn.Module):
    def __init__(self, n_classes=1):
        super(MicrobiomeTabularConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(19360, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
