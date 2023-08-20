import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, ip_size, h_size, no_classes):
        super(NN, self).__init__()
        self.l1 = nn.Linear(ip_size, h_size)
        self.l2 = nn.Linear(h_size, h_size)
        self.l3 = nn.Linear(h_size, no_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)

        return out