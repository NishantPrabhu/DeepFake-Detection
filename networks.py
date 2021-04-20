
""" 
Network definitions.
Modified output layers of few models.
"""

import torch 
import torch.nn as nn 
import torchvision.models as models


class Resnet34(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        net = models.resnet34(pretrained=pretrained)
        layers = list(net.children()) 
        self.base = nn.Sequential(*layers[:-1]) 
        self.flatten = nn.Flatten(dim=-1)
        self.classifier = nn.Linear(in_features=512, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, forward="full"):
        if forward == "full":
            x = self.base(x)
            x = self.flatten(x)
            x = self.classifier(x)
            x = self.log_softmax(x)
        elif forward == "base":
            x = self.base(x)
            x = self.flatten(x)
        return x


class Resnet50(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        net = models.resnet50(pretrained=pretrained)
        layers = list(net.children()) 
        self.base = nn.Sequential(*layers[:-1]) 
        self.flatten = nn.Flatten(dim=-1)
        self.classifier = nn.Linear(in_features=2048, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, forward="full"):
        if forward == "full":
            x = self.base(x)
            x = self.flatten(x)
            x = self.classifier(x)
            x = self.log_softmax(x)
        elif forward == "base":
            x = self.base(x)
            x = self.flatten(x)
        return x


class Resnet101(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        net = models.resnet101(pretrained=pretrained)
        layers = list(net.children()) 
        self.base = nn.Sequential(*layers[:-1]) 
        self.flatten = nn.Flatten(dim=-1)
        self.classifier = nn.Linear(in_features=2048, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, forward="full"):
        if forward == "full":
            x = self.base(x)
            x = self.flatten(x)
            x = self.classifier(x)
            x = self.log_softmax(x)
        elif forward == "base":
            x = self.base(x)
            x = self.flatten(x)
        return x


class Resnet152(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        net = models.resnet152(pretrained=pretrained)
        layers = list(net.children()) 
        self.base = nn.Sequential(*layers[:-1]) 
        self.flatten = nn.Flatten(dim=-1)
        self.classifier = nn.Linear(in_features=2048, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, forward="full"):
        if forward == "full":
            x = self.base(x)
            x = self.flatten(x)
            x = self.classifier(x)
            x = self.log_softmax(x)
        elif forward == "base":
            x = self.base(x)
            x = self.flatten(x)
        return x


class EfficientNet(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        self.base = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b3', pretrained=pretrained)
        self.base.classifier = nn.Linear(in_features=1536, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.base(x)
        x = self.log_softmax(x)
        return x