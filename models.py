import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, layers, m):
        super().__init__()
        net = []
        net.append(nn.Linear(in_dim, hidden))
        net.append(nn.ReLU())

        for _ in range(layers - 2):
            net.append(nn.Linear(hidden, hidden))
            net.append(nn.ReLU())

        net.append(nn.Linear(hidden, m))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
