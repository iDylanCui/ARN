import torch
import torch.nn as nn
import torch.nn.functional as F

class DistMult(nn.Module):
    def __init__(self):
        super(DistMult, self).__init__()

    def forward(self, E1, R, E2):
        S = torch.mm(E1 * R, E2.transpose(1, 0))
        S = F.sigmoid(S)
        return S

    def forward_fact(self, E1, R, E2):
        S = torch.sum(E1 * R * E2, dim=1, keepdim=True)
        S = F.sigmoid(S)
        return S