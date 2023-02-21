###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# coupling_detection.py:

###############################################################################
# Dependencies
###############################################################################

import numpy as np
import random
import torch
from torch import nn
from torch.autograd import functional as F
import matplotlib.pyplot as plt

from expressions import *
from integrators import *
from train import *

###############################################################################
# Main Function
###############################################################################

def detect_coupling(X, y, left_vars=[], right_vars=[], use_gpu=True):
    if (use_gpu and torch.cuda.is_available()):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    X_train = torch.cat((X[:100, :], X[120:200, :]), dim=0)
    X_test = X[110:120, :]
    y_train = torch.cat((y[:100, :], y[120:200, :]), dim=0)
    y_test = y[110:120, :]

    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    left_terms = [[0, 1, 2, 3], [0, 1, 4, 5], [2, 3, 4, 5]]
    right_terms = [[0], [1], [2], [3], [4], [5]]

    print("Euclidean")
    coupling_detector = CustomSymplectic(left_terms, right_terms, device, num_steps=10, single_net_right=True, single_net_left=True, type=5).to(device)
    coupling_detector.train(X_train, y_train, X_test, y_test, print_every=700)
    print("Manhattan")
    coupling_detector = CustomSymplectic(left_terms, right_terms, device, num_steps=10, single_net_right=True, single_net_left=True, type=4).to(device)
    coupling_detector.train(X_train, y_train, X_test, y_test, print_every=700)

class CustomSymplectic(nn.Module):
    def __init__(self, left_terms, right_terms, device, dropout=0.0, hidden_size=128, hidden_layers=8, step_size=0.1, num_steps=1, single_net_left=False, single_net_right=False, type=0):
        super(CustomSymplectic, self).__init__()

        # Save meta information
        self.left_terms = left_terms
        self.right_terms = right_terms
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.device = device
        self.single_net_left = single_net_left
        self.single_net_right = single_net_right
        self.type = type

        # Set integrator and integration details
        self.integrator = fourth_order
        self.step_size = step_size
        self.num_steps = num_steps
        self.t0 = torch.Tensor([0.0]).to(self.device)
        self.t1 = torch.Tensor([self.step_size]).to(self.device)

        # Construct ensemble of methods for left side
        if (self.type == 0):
            self.left_models_list = [self.get_model(len(x)).to(self.device) for x in self.left_terms]
        else:
            self.left_models_list = [self.get_model(1).to(self.device) for x in self.left_terms]
        self.left_side = SeparabilityWrapperLeft(self.left_models_list, torch.Tensor(self.left_terms).long().to(self.device), type=type, single_net=single_net_left, passed_indices=torch.Tensor([0, 1, 2]).long().to(device)).to(self.device)
        self.left_side_grad = GradientWrapper(self.left_side).to(self.device)

        # Construct ensemble of methods for right side
        self.right_models_list = [self.get_model(len(x)).to(self.device) for x in self.right_terms]
        self.right_side = SeparabilityWrapperRight(self.right_models_list, torch.Tensor(self.right_terms).long().to(self.device), single_net=single_net_right).to(self.device)
        self.right_side_grad = GradientWrapper(self.right_side).to(self.device)

    def forward(self, X):
        # Separate out X into left and right side; track gradients for numerical integration
        num_vars = len(X[0])
        q0 = X[:, :num_vars//2]
        p0 = X[:, num_vars//2:]
        q0 = Variable(q0, requires_grad=True)
        p0 = Variable(p0, requires_grad=True)

        # Integrate for appropriate number of steps
        for i in range(self.num_steps):
            p, q = self.integrator(p0, q0, self.t0, self.t1, self.right_side_grad, self.left_side_grad)
            p0, q0 = p, q

        # Zip variables back up and return
        y = torch.cat((q, p), axis=1)[:, :]
        return y

    def train(self, X_train, y_train, X_test, y_test, n_epochs=2810, lr=0.0005, print_every=100, early_stop_value=1.00):
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        max_reward = 0
        max_test_reward = 0
        for i in range(n_epochs):
            optim.zero_grad()
            y_pred = self(X_train)
            net_loss = -1 * reward_nrmse_grad(y_pred, y_train)
            max_reward = max(max_reward, abs(net_loss.item()))

            # Compute test reward
            test_reward = abs(reward_nrmse_grad(self(X_test), y_test).item())
            max_test_reward = max(max_test_reward, test_reward)
            if (abs(test_reward) > early_stop_value):
                break
            if (i % print_every == 0):
                print("Epoch {}:".format(i+1))
                print("\tCurrent Train Reward : {}".format(abs(net_loss.item())))
                print("\tMaximum Train Reward : {}".format(max_reward))
                print("\tMaximum Test Reward : {}".format(max_test_reward))

            net_loss.backward()
            optim.step()

    def get_model(self, num_vars):
        input_layer = [
            nn.Linear(num_vars, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout)
        ]
        hidden_layers = [
            [nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout)] for i in range(self.hidden_layers-1)
        ]
        hidden_layers = [item for sublist in hidden_layers for item in sublist]
        output_layer = [nn.Linear(self.hidden_size, 1)]
        combined_layers = input_layer + hidden_layers + output_layer
        return nn.Sequential(*combined_layers)

    def __str__(self):
        return '[' + str(self.Vq) + '] + [' + str(self.Tp) + ']'

class SeparabilityWrapperLeft(nn.Module):
    def __init__(self, model_list, indices_list, type='manhattan', single_net=False, passed_indices=None):
        super(SeparabilityWrapperLeft, self).__init__()
        self.models = torch.nn.ModuleList(model_list)
        self.indices_list = indices_list
        self.type = type
        self.single_net = single_net
        if (self.type == 4 or self.type == 5):
            self.indices_list = passed_indices
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        if (self.type == 0): # Normal
            X_in = X
        elif (self.type == 1): # Sum
            X_in = torch.sum(X, axis=-1)[:, None]
        elif (self.type == 2): # Product
            X_in = torch.prod(X, axis=-1)[:, None]
        elif (self.type == 3): # Product them sum
            X_in = ((X[:, 2] * X[:, 0]) + (X[:, 3]*X[:, 1]))[:, None]
        elif (self.type == 4): # Manhattan
            X_in_1 = (torch.abs(X[:, 2] - X[:, 0]) + torch.abs(X[:, 3] - X[:, 1]))
            X_in_2 = (torch.abs(X[:, 4] - X[:, 2]) + torch.abs(X[:, 5] - X[:, 3]))
            X_in_3 = (torch.abs(X[:, 4] - X[:, 0]) + torch.abs(X[:, 5] - X[:, 1]))
            X_in = torch.stack((X_in_1, X_in_2, X_in_3))
        elif (self.type == 5): # Euclidean
            X_in_1 = torch.sqrt((X[:, 2] - X[:, 0])**2 + (X[:, 3] - X[:, 1])**2)
            X_in_2 = torch.sqrt((X[:, 4] - X[:, 2])**2 + (X[:, 5] - X[:, 3])**2)
            X_in_3 = torch.sqrt((X[:, 4] - X[:, 0])**2 + (X[:, 5] - X[:, 1])**2)
            X_in = torch.stack((X_in_1, X_in_2, X_in_3))

        if (self.single_net):
            if (self.type == 0):
                futures = [torch.jit.fork(self.models[0], torch.index_select(X_in, 1, self.indices_list[i])) for i, model in enumerate(self.models)]
            elif (self.type == 4 or self.type == 5):
                futures = [torch.jit.fork(model, torch.index_select(X_in, 0, self.indices_list[i])[0, :, None]) for i, model in enumerate(self.models)]
            else:
                futures = [torch.jit.fork(self.models[0], X_in) for i, model in enumerate(self.models)]
        else:
            if (self.type == 0):
                futures = [torch.jit.fork(model, torch.index_select(X_in, 1, self.indices_list[i])) for i, model in enumerate(self.models)]
            else:
                futures = [torch.jit.fork(model, X_in) for i, model in enumerate(self.models)]
        results = [torch.jit.wait(fut) for fut in futures]
        results = torch.stack(results, dim=0).sum(dim=0)
        return results

class SeparabilityWrapperRight(nn.Module):
    def __init__(self, model_list, indices_list, single_net=False):
        super(SeparabilityWrapperRight, self).__init__()
        self.models = torch.nn.ModuleList(model_list)
        self.indices_list = indices_list
        self.single_net = single_net
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        if (self.single_net):
            futures = [torch.jit.fork(self.models[0], torch.index_select(X, 1, self.indices_list[i])) for i, model in enumerate(self.models)]
        else:
            futures = [torch.jit.fork(model, torch.index_select(X, 1, self.indices_list[i])) for i, model in enumerate(self.models)]
        results = [torch.jit.wait(fut) for fut in futures]
        results = torch.stack(results, dim=0).sum(dim=0)
        return results

# # Hessian regularization
# hessian_regularization = 0.0
# # for entry in X_train:
# #     left_hess = F.hessian(self.left_side, Variable(entry[0:2], requires_grad=True), create_graph=True)
#     # hessian_regularization += torch.abs(left_hess.sum() - left_hess.trace())
#     # right_hess = F.hessian(self.right_side, Variable(entry[2:4], requires_grad=True), create_graph=True)
#     # hessian_regularization += torch.abs(right_hess.sum() - right_hess.trace())
# # print("{} | {} | {}".format(test_reward, max_test_reward, hessian_regularization.item()))
