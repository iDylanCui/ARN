import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TuckER(nn.Module):
    def __init__(self, args):
        super(TuckER, self).__init__()
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        self.register_parameter('W', nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.relation_dim, self.entity_dim, self.entity_dim)), dtype=torch.float, device="cuda", requires_grad=True)))
        self.input_dropout = torch.nn.Dropout(0.3)
        self.hidden_dropout1 = torch.nn.Dropout(0.4)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)

        self.bn0 = torch.nn.BatchNorm1d(self.entity_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.entity_dim)

    def forward(self, E1, R, E2): 
        x = self.bn0(E1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, E1.size(1)) 

        W_mat = torch.mm(R, self.W.view(R.size(1), -1)) 
        W_mat = W_mat.view(-1, E1.size(1), E1.size(1)) 
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, E1.size(1)) 
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, E2.transpose(1,0)) 
        pred = torch.sigmoid(x)
        return pred

    def forward_fact(self, E1, R, E2):
        x = self.bn0(E1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, E1.size(1)) 

        W_mat = torch.mm(R, self.W.view(R.size(1), -1)) 
        W_mat = W_mat.view(-1, E1.size(1), E1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, E1.size(1))
        x = self.bn1(x)
        X = self.hidden_dropout2(x) 

        X = torch.matmul(X.unsqueeze(1), E2.unsqueeze(2)).squeeze(2)
        S = torch.sigmoid(X)
        return S