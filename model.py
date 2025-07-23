import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    def __init__(self, inChannels, outChannels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)

        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
class ResNet32(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.inChannels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #3 earlier
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        self.layer1 = self.makeLayer(64, 3, stride=1)
        self.layer2 = self.makeLayer(128, 4, stride=2)
        self.layer3 = self.makeLayer(256, 5, stride=2)
        self.layer4 = self.makeLayer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def makeLayer(self, outChannels, blocks, stride):
        downsample = None
        if stride != 1 or self.inChannels != outChannels:
            downsample = nn.Sequential(nn.Conv2d(self.inChannels, outChannels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(outChannels))
        
        layers = []
        layers.append(BasicBlock(self.inChannels, outChannels, stride, downsample))
        self.inChannels = outChannels

        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inChannels, outChannels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x