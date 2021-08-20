import torch
import torch.nn as nn

class TransE(nn.Module):
    def __init__(self):
        super(TransE, self).__init__()

    def forward_train(self, E1, R, E2):
        return torch.abs(E1 + R - E2) 

    def forward(self, E1, R, E2): 
        size_e1 = E1.size()
        size_e2 = E2.size()

        A = torch.sum((E1 + R) * (E1 + R), dim=1)
        B = torch.sum(E2 * E2, dim=1)
        AB = torch.mm((E1 + R), E2.transpose(1, 0))
        S = A.view(size_e1[0], 1) + B.view(1, size_e2[0]) - 2 * AB
        
        return torch.sigmoid(-torch.sqrt(S)) 

    def forward_fact(self, E1, R, E2):
        return torch.sigmoid(-torch.sqrt(torch.sum((E1 + R - E2) * (E1 + R - E2), dim=1, keepdim=True)))
