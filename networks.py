
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
        self.base = models.resnet34(pretrained=pretrained) 
        self.base.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.base(x)
        x = self.log_softmax(x)
        return x


class Resnet50(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        self.base = models.resnet50(pretrained=pretrained) 
        self.base.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.base(x)
        x = self.log_softmax(x)
        return x


class Resnet101(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        self.base = models.resnet101(pretrained=pretrained)
        self.base.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.base(x)
        x = self.log_softmax(x)
        return x


class Resnet152(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        self.base = models.resnet152(pretrained=pretrained)
        self.base.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.base(x)
        x = self.log_softmax(x)
        return x


class Densenet121(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        self.base = models.densenet121(pretrained=pretrained)
        self.base.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.base(x)
        x = self.log_softmax(x)
        return x


class Densenet161(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        self.base = models.densenet161(pretrained=pretrained)
        self.base.classifier = nn.Linear(in_features=2208, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.base(x)
        x = self.log_softmax(x)
        return x


class Densenet169(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        self.base = models.densenet169(pretrained=pretrained)
        self.base.classifier = nn.Linear(in_features=1664, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.base(x)
        x = self.log_softmax(x)
        return x


class Densenet201(nn.Module):

    def __init__(self, pretrained=False):
        super().__init__()
        self.base = models.densenet201(pretrained=pretrained)
        self.base.classifier = nn.Linear(in_features=1920, out_features=2, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.base(x)
        x = self.log_softmax(x)
        return x