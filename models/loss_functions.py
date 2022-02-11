import torch

def recon_loss(pos_edge_proba, neg_edge_proba):
    EPS = 1e-15
    pos_loss = -torch.log(pos_edge_proba + EPS).mean()

    neg_loss = -torch.log(1 - neg_edge_proba + EPS).mean()

    return pos_loss + neg_loss