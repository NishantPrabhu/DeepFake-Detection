
""" 
Custom definitions of loss functions
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class LogLoss(nn.Module):
    """ Allows smooth label functionality for mixup """

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        loss = (target * torch.log(output[:, 1])) + ((1 - target) * torch.log(output[:, 0]))
        return -loss.mean() 



if __name__ == "__main__":

    targets = torch.tensor([1, 0, 0, 1])
    outputs = torch.tensor([[0.02, 0.98], [0.92, 0.08], [0.99, 0.01], [0.05, 0.95]])

    loss = LogLoss()
    print(loss(outputs, targets))