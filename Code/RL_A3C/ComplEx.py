import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplEx(nn.Module):
    def __init__(self):
        super(ComplEx, self).__init__()

    def forward(self, E1_real, R_real, E2_real, E1_img, R_img, E2_img):
        def dist_mult(E1, R, E2):
            return torch.mm(E1 * R, E2.transpose(1, 0))

        rrr = dist_mult(R_real, E1_real, E2_real)
        rii = dist_mult(R_real, E1_img, E2_img)
        iri = dist_mult(R_img, E1_real, E2_img)
        iir = dist_mult(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir
        S = F.sigmoid(S)
        return S

    def forward_fact(self, E1_real, R_real, E2_real, E1_img, R_img, E2_img):
        def dist_mult_fact(E1, R, E2):
            return torch.sum(E1 * R * E2, dim=1, keepdim=True)

        rrr = dist_mult_fact(R_real, E1_real, E2_real)
        rii = dist_mult_fact(R_real, E1_img, E2_img)
        iri = dist_mult_fact(R_img, E1_real, E2_img)
        iir = dist_mult_fact(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir
        S = F.sigmoid(S)
        return S