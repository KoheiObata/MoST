import torch
import torch.nn.functional as F
import random

def ex_range(start, end, excluded_number):
    return [x for x in range(start, end) if x != excluded_number]

def contrastive_loss(z1,z2, alpha=1):
    # z1: B x T x Ch
    # z2: B x T x Ch
    b,t,Ch=z1.size()
    halfCh = int(Ch/2)

    z1_new = torch.stack([z1[:,:,:halfCh],z1[:,:,halfCh:]], dim=0) # 2 x B x T x Ch
    z2_new = torch.stack([z2[:,:,:halfCh],z2[:,:,halfCh:]], dim=0)
    z1_new = torch.reshape(z1_new, (2*b, t, halfCh)) # (2xB) x T x Ch
    z2_new = torch.reshape(z2_new, (2*b, t, halfCh))

    loss = instance_loss(z1[:,:,:halfCh],z2[:,:,:halfCh])
    loss += instance_loss(z1[:,:,halfCh:],z2[:,:,halfCh:])
    loss += alpha*mode_loss(z1[:,:,halfCh:],z1[:,:,:halfCh])
    return loss


def instance_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def mode_loss(z1,z2):
    # z1: B x T x halfCh
    # z2: B x T x halfCh same d, diff aug
    b,t,Ch=z1.size()
    k_pos = 1
    k_neg = b-1

    sim_pos = torch.matmul(z1.permute(1,0,2), z2.permute(1,2,0))
    sim_neg = torch.matmul(z1.permute(1,0,2), z2.permute(1,2,0))
    sim = torch.zeros((t,b,k_pos+k_neg))
    for i in range(b):
        pos_indices = torch.tensor(random.sample(range(i, i+1), 1)) #random pick positive sample except diagonal element
        pos_indices = torch.tensor(pos_indices)
        neg_indices = torch.tensor(random.sample(ex_range(0, b, i), k_neg)) #random pick negative samples
        neg_indices = torch.tensor(neg_indices)
        sim[:,i,:] = torch.cat([sim_pos[:,i,pos_indices], sim_neg[:,i,neg_indices]], dim=-1)
    logits = -F.log_softmax(sim, dim=-1)
    loss = logits[:,:,0].mean()
    return loss