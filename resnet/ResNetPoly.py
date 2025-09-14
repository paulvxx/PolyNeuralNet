import torch
import torch.nn as nn
import torch.nn.functional as F
from PolynomialActivation import Polynomial

# Basic building block (Polynomial)
class BasicPolyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, degree=2, initialization='linear'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            )
        
        self.poly = None
        # Coefficient Initialization is the identity function f(x) = x
        if initialization == 'identity'
          ctensor = torch.zeros(degree + 1)
          ctensor[-2] = 1.0
          self.poly = [Polynomial(ctensor), Polynomial(ctensor)]
        # Coefficient Initialization is Uniform Random on [0,1]
        else:
          self.poly = [Polynomial(torch.rand(degree + 1)), Polynomial(torch.rand(degree + 1))]
        
    def forward(self, x):
        out = self.poly[1](self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return self.poly[2](out)


# ResNet20 architecture
class ResNet20(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, degree=2, initialization='identity'):
        super().__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)

        # Polynomial Function
        # Random initialization, uniform on [0, 1]
        self.polynomial = Polynomial(torch.rand(degree + 1))
        # f(x) = x
        if (initialization=='identity'):
            ctensor = torch.zeros(degree + 1)
            ctensor[-2] = 1.0
            self.polynomial = Polynomial(ctensor)    
        
        self.layer1 = nn.Sequential(
            BasicPolyBlock(16, 16, stride=1),
            BasicPolyBlock(16, 16, stride=1),
            BasicPolyBlock(16, 16, stride=1)
        )
        self.layer2 = nn.Sequential(
            BasicPolyBlock(16, 32, stride=2),
            BasicPolyBlock(32, 32, stride=1),
            BasicPolyBlock(32, 32, stride=1)
        )
        self.layer3 = nn.Sequential(
            BasicPolyBlock(32, 64, stride=2),
            BasicPolyBlock(64, 64, stride=1),
            BasicPolyBlock(64, 64, stride=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.polynomial(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
