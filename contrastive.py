import torch
import torch.nn as nn
class Contrast_L(nn.Module):
    def __init__(self, tau):
        super(Contrast_L, self).__init__()
        self.tau = tau

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, x_0, x_1, L, gamma):
        sim = self.sim(x_0, x_1)
        l = sim / (torch.sum(sim, dim=1).view(-1, 1) + 1e-8) #back
        pos = torch.eye(len(L))
        back = l.mul(pos).sum(dim=-1)
        front = (sim.mul(pos).sum(dim=-1))**2/(L ** gamma)
        loss = -torch.log(front*back.sum(dim=-1)).mean()
        return loss

class Contrast_gene(nn.Module):
    def __init__(self, tau):
        super(Contrast_gene, self).__init__()
        self.tau = tau

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, x_r, x_pos, x_neg):
        pos_score = self.sim(x_r, x_pos)
        neg_score = self.sim(x_r, x_neg)
        pos = torch.eye(len(x_r))
        ss = pos_score.mul(pos).sum(dim=-1)/(neg_score.mul(pos).sum(dim=-1)+ 1e-8)
        loss = -torch.log(ss.sum(dim=-1)).mean()
        return loss
