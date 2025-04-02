# vgg16_pruning/model.py
import torch
import torch.nn as nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(VGG16, self).__init__()
        base_model = models.vgg16(pretrained=pretrained)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_conv_layers(self):
        return [m for m in self.features if isinstance(m, nn.Conv2d)]
