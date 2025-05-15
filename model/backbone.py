import torch
import torch.nn as nn
from torchvision import models

class USFeatureExtractor(nn.Module):
    def __init__(self, freeze_feat=True, output_dim=512):
        super().__init__()
        self.base = models.resnet50(pretrained=True)
        if freeze_feat:
            for param in self.base.parameters():
                param.requires_grad = False
        self.base.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, output_dim)
        )
        self.output_dim = output_dim

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        return self.base.fc(x)

class ClinicalEncoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.fc(x)
    