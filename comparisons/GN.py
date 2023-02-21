import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch_geometric.data import Data, DataLoader
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus
from torch.autograd import Variable, grad

from data_loading import *

###############################################################################
# Main Training Loop
###############################################################################

def main():
    # 3 body problem
    dataset = load_hamiltonian_dataset('.././datasets/3body/3body_noise_3.pkl')[:40]
    X, y = separate_xy(dataset, num_steps=1)
    q_i_X = X[:, 0:2]
    q_j_X = X[:, 2:4]
    q_k_X = X[:, 4:6]
    p_i_X = X[:, 6:8]
    p_j_X = X[:, 8:10]
    p_k_X = X[:, 10:12]
    particle_i = torch.cat((q_i_X, p_i_X), axis=1)
    particle_j = torch.cat((q_j_X, p_j_X), axis=1)
    particle_k = torch.cat((q_k_X, p_k_X), axis=1)
    X_new = torch.cat((particle_i[:, None], particle_j[:, None], particle_k[:, None]), axis=1)

    n = 3
    dim = 4
    edge_index = get_edge_index(n, '3body')


    hgn = HGN(dim, n).cuda()
    input = Variable(X_new[0].cuda(), requires_grad=True)
    out = hgn(input, edge_index.cuda()).sum()
    print(hgn.message(X_new[0,0, None].cuda(), X_new[0,1,None].cuda()))

    # print(grad(out, input))
    # g = type('obj', (object,), {'x' : X_new.cuda(), 'edge_index': edge_index.cuda()})
    # print(hgn.just_derivative(g))
    # print(hgn(X_new[0].cuda(), edge_index.cuda()).sum())

###############################################################################
# Graph Network Models
###############################################################################

def make_packer(n, n_f):
    def pack(x):
        return x.reshape(-1, n_f*n)
    return pack

def make_unpacker(n, n_f):
    def unpack(x):
        return x.reshape(-1, n, n_f)
    return unpack

def get_edge_index(n, sim):
    if sim in ['string', 'string_ball']:
        #Should just be along it.
        top = torch.arange(0, n-1)
        bottom = torch.arange(1, n)
        edge_index = torch.cat(
            (torch.cat((top, bottom))[None],
             torch.cat((bottom, top))[None]), dim=0
        )
    else:
        adj = (np.ones((n, n)) - np.eye(n)).astype(int)
        edge_index = torch.from_numpy(np.array(np.where(adj)))

    return edge_index

class HGN(MessagePassing):
    def __init__(self, n_f, ndim, hidden=300):
        super(HGN, self).__init__(aggr='add')  # "Add" aggregation.
        self.pair_energy = Seq(
            Lin(2*n_f, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, 1)
        )

        self.self_energy = Seq(
            Lin(n_f, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, 1)
        )
        self.ndim = ndim

    def forward(self, x, edge_index):
        #x is [n, n_f]
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.pair_energy(tmp)

    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]

        sum_pair_energies = aggr_out
        self_energies = self.self_energy(x)
        return sum_pair_energies + self_energies

    def just_derivative(self, g, augment=False, augmentation=3):
        #x is [n, n_f]f
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim)*augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)

        #Make momenta:
        x = Variable(torch.cat((x[:, :ndim], x[:, ndim:2*ndim]*x[:, [-1]*ndim], x[:, 2*ndim:]), dim=1), requires_grad=True)

        edge_index = g.edge_index
        total_energy = self.propagate(
                edge_index, size=(x.size(0), x.size(0)),
                x=x).sum()

        dH = grad(total_energy, x, create_graph=True)[0]
        dH_dq = dH[:, :ndim]
        dH_dp = dH[:, ndim:2*ndim]

        dq_dt = dH_dp
        dp_dt = -dH_dq
        dv_dt = dp_dt/x[:, [-1]*ndim]
        return torch.cat((dq_dt, dv_dt), dim=1)

    def loss(self, g, augment=True, square=False, reg=True, augmentation=3, **kwargs):
        all_derivatives = self.just_derivative(g, augment=augment, augmentation=augmentation)
        ndim = self.ndim
        dv_dt = all_derivatives[:, self.ndim:]

        if reg:
            ## If predicting dq_dt too, the following regularization is important:
            edge_index = g.edge_index
            x = g.x
            #make momenta:
            x = Variable(torch.cat((x[:, :ndim], x[:, ndim:2*ndim]*x[:, [-1]*ndim], x[:, 2*ndim:]), dim=1), requires_grad=True)
            self_energies = self.self_energy(x)
            total_energy = self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)
            #pair_energies = total_energy - self_energies
            #regularization = 1e-3 * torch.sum((pair_energies)**2)
            dH = grad(total_energy.sum(), x, create_graph=True)[0]
            dH_dother = dH[2*ndim:]
            #Punish total energy and gradient with respect to other variables:
            regularization = 1e-6 * (torch.sum((total_energy)**2) + torch.sum((dH_dother)**2))
            return torch.sum(torch.abs(g.y - dv_dt)) + regularization
        else:
            return torch.sum(torch.abs(g.y - dv_dt))
        #return torch.sum(torch.abs(g.y - dv_dt))

if __name__=='__main__':
    main()
