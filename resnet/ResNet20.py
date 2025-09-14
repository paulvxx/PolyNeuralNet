import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic building block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# ResNet20 architecture
class ResNet20(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1),
            BasicBlock(16, 16, stride=1)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(16, 32, stride=2),
            BasicBlock(32, 32, stride=1),
            BasicBlock(32, 32, stride=1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
