import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional
import numpy as np
import math
import random
import pickle

import numpy as np
import pysindy as ps

from data_loading import *

def main():
    # Parameters
    lr = 10e-4
    n_epochs = 100
    regularization = 10e-3

    # ~ Set GPU/CPU if desired ~
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.FloatTensor)

    # Select Dataset and corresponding Tp/Vq to use
    with torch.no_grad():
        # Harmonic
        # dataset = load_hamiltonian_dataset('.././datasets/harmonic_constant/harmonic_constant_noise_1.pkl')
        # Tp = univariate_trinomial().to(device)
        # Vq = univariate_trinomial().to(device)
        # X, y = separate_xy(dataset, num_steps=1)

        # Pendulum
        # dataset = load_hamiltonian_dataset('.././datasets/pendulum/pendulum_noise_3.pkl')
        # Tp = univariate_trinomial().to(device)
        # Vq = univariate_trigonometric_term().to(device)
        # X, y = separate_xy(dataset, num_steps=1)

        # Kepler Problem
        # dataset = load_hamiltonian_dataset('.././datasets/kepler/kepler_noise_3.pkl')
        # Tp = fourvariate_poly(degree=2).to(device)
        # Vq = fourvariate_poly(degree=2).to(device)
        # X, y = separate_xy(dataset, num_steps=1)

        # Three Body Problem
        dataset = load_hamiltonian_dataset('.././datasets/3body/3body_noise_3.pkl')
        Tp = sixvariate_poly(degree=2).to(device)
        Vq = sixvariate_poly(degree=2).to(device)
        X, y = separate_xy(dataset, num_steps=1)

    # Set Vq and Tp
    _Tp = gradient_wrapper(Tp)
    _Vq = gradient_wrapper(Vq)
    model = SSINN(_Tp, _Vq, fourth_order, tol=1e-2).to(device)

    # ~ Train Model ~
    train_model(model, X, y, lr, n_epochs, regularization, device)

def train_model(model, X, y, lr, n_epochs, regularization, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    p_criterion = nn.L1Loss().to(device)
    q_criterion = nn.L1Loss().to(device)
    lr_decay = {70: 0.1, 80: 0.01, 90: 0.001}
    dt = 0.1

    t0 = torch.Tensor([0])
    t1 = torch.Tensor([dt])

    # Iterate over each epoch
    for j in range(n_epochs):

        # Decay learning rate if desired
        if (j in lr_decay):
            print("learning rate decayed to " + str(lr*lr_decay[j]))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr*lr_decay[j]

        predicted = []
        for i in range(len(X)):
            optimizer.zero_grad()
            q0 = Variable(X[i,:len(X[0])//2, None], requires_grad=True)
            p0 = Variable(X[i,len(X[0])//2:, None], requires_grad=True)
            q1 = Variable(y[i,:len(X[0])//2, None], requires_grad=True)
            p1 = Variable(y[i,len(X[0])//2:, None], requires_grad=True)
            p1_pred, q1_pred = model(p0, q0, t0, t1)
            predicted.append(torch.cat((q1_pred[:,0], p1_pred[:,0])).tolist())

            # Accumulate losses and optimize
            l1_regularization = 0.
            for param in model.parameters():
                l1_regularization += param.abs().sum()
            loss = p_criterion(p1_pred, p1) + q_criterion(p1_pred, q1) + regularization*l1_regularization
            if (not torch.isnan(loss).item()):
                loss.backward()
                optimizer.step()
                scheduler.step()
        one_step_rew = round(reward_nrmse(torch.Tensor(predicted), y), 3)

        # Predict entire trajectory and compute reward
        predicted = []
        q0 = Variable(X[0,:len(X[0])//2, None], requires_grad=True)
        p0 = Variable(X[0,len(X[0])//2:, None], requires_grad=True)
        for i in range(len(X)):
            optimizer.zero_grad()
            p1, q1 = model(p0, q0, t0, t1)
            predicted.append(torch.cat((q1_pred[:,0], p1_pred[:,0])).tolist())
            p0, q0 = p1, q1
        traject_reward = round(reward_nrmse(torch.Tensor(predicted), y), 3)
        print(one_step_rew, traject_reward)

###############################################################################
# FUNCTION SPACES
###############################################################################

import torch
from torch import nn

# Used to wrap each function space so that the forward pass obtains the gradient
class gradient_wrapper(nn.Module):
    def __init__(self, space):
        super(gradient_wrapper, self).__init__()
        self.space = space

    def forward(self, x):
        return torch.autograd.grad(self.space(x), x, create_graph=True)[0]

###############################################################################
# Used for Harmonic Oscillator
###############################################################################

# Bivariate polynomial function space. For use with a single point in 2D space.
class bivariate_poly(nn.Module):
    def __init__(self, degree=3):
        super(bivariate_poly, self).__init__()
        self.degree = degree
        self.fc1 = nn.Linear(degree**2 + 2*degree, 1, bias=False)

    def forward(self, x):
        xc = torch.ones([self.degree])*(x[:,0])
        yc = torch.ones([self.degree])*(x[:,1])
        xc_pow = torch.pow(xc, torch.Tensor([i for i in range(1, self.degree+1)]))
        yc_pow = torch.pow(yc, torch.Tensor([i for i in range(1, self.degree+1)]))
        combos = torch.ger(xc_pow, yc_pow).flatten()
        input = torch.cat((xc_pow, yc_pow, combos))
        out = self.fc1(input)
        return out

###############################################################################
# Used for Kepler
###############################################################################

# Polynomial space with 5 input variables. Only supports degree 2, 3, 4, 6, or 10.
class fourvariate_poly(nn.Module):
    def __init__(self, degree):
        super(fourvariate_poly, self).__init__()
        self.degree = degree
        if (degree == 2):
            self.fc1 = nn.Linear(32, 1, bias=False)
        elif (degree == 3):
            self.fc1 = nn.Linear(66, 1, bias=False)
        elif (degree == 4):
            self.fc1 = nn.Linear(180, 1, bias=False)
        elif (degree == 6):
            self.fc1 = nn.Linear(390, 1, bias=False)
        elif (degree == 10):
            self.fc1 = nn.Linear(1050, 1, bias=False)

    def forward(self, x):
        x1 = torch.ones([self.degree])*(x[:,0][0])
        x2 = torch.ones([self.degree])*(x[:,0][1])
        x3 = torch.ones([self.degree])*(x[:,0][2])
        x4 = torch.ones([self.degree])*(x[:,0][3])
        x1_pow = torch.pow(x1, torch.Tensor([i for i in range(1, self.degree+1)]))
        x2_pow = torch.pow(x2, torch.Tensor([i for i in range(1, self.degree+1)]))
        x3_pow = torch.pow(x3, torch.Tensor([i for i in range(1, self.degree+1)]))
        x4_pow = torch.pow(x4, torch.Tensor([i for i in range(1, self.degree+1)]))
        x1x2 = torch.ger(x1_pow, x2_pow).flatten()
        x1x3 = torch.ger(x1_pow, x3_pow).flatten()
        x1x4 = torch.ger(x1_pow, x4_pow).flatten()
        x2x3 = torch.ger(x2_pow, x3_pow).flatten()
        x2x4 = torch.ger(x2_pow, x4_pow).flatten()
        x3x4 = torch.ger(x3_pow, x4_pow).flatten()
        input = torch.cat((x1_pow, x2_pow, x3_pow, x4_pow,
                           x1x2, x1x3, x1x4,
                           x2x3, x2x4,
                           x3x4))
        out = self.fc1(input)
        return out

###############################################################################
# Used for Three-Body
###############################################################################

class sixvariate_poly(nn.Module):
    def __init__(self, degree):
        super(sixvariate_poly, self).__init__()
        self.degree = degree
        if (degree == 2):
            self.fc1 = nn.Linear(76, 1, bias=False)
        elif (degree == 3):
            self.fc1 = nn.Linear(105, 1, bias=False)
        elif (degree == 4):
            self.fc1 = nn.Linear(180, 1, bias=False)
        elif (degree == 6):
            self.fc1 = nn.Linear(390, 1, bias=False)
        elif (degree == 10):
            self.fc1 = nn.Linear(1050, 1, bias=False)

    def forward(self, x):
        x1 = torch.ones([self.degree])*(x[:,0][0])
        x2 = torch.ones([self.degree])*(x[:,0][1])
        x3 = torch.ones([self.degree])*(x[:,0][2])
        x4 = torch.ones([self.degree])*(x[:,0][3])
        x5 = torch.ones([self.degree])*(x[:,0][4])
        x6 = torch.ones([self.degree])*(x[:,0][5])
        x1_pow = torch.pow(x1, torch.Tensor([i for i in range(1, self.degree+1)]))
        x2_pow = torch.pow(x2, torch.Tensor([i for i in range(1, self.degree+1)]))
        x3_pow = torch.pow(x3, torch.Tensor([i for i in range(1, self.degree+1)]))
        x4_pow = torch.pow(x4, torch.Tensor([i for i in range(1, self.degree+1)]))
        x5_pow = torch.pow(x5, torch.Tensor([i for i in range(1, self.degree+1)]))
        x6_pow = torch.pow(x6, torch.Tensor([i for i in range(1, self.degree+1)]))
        x1x2 = torch.ger(x1_pow, x2_pow).flatten()
        x1x3 = torch.ger(x1_pow, x3_pow).flatten()
        x1x4 = torch.ger(x1_pow, x4_pow).flatten()
        x1x5 = torch.ger(x1_pow, x5_pow).flatten()
        x1x6 = torch.ger(x1_pow, x6_pow).flatten()
        x2x3 = torch.ger(x2_pow, x3_pow).flatten()
        x2x4 = torch.ger(x2_pow, x4_pow).flatten()
        x2x5 = torch.ger(x2_pow, x5_pow).flatten()
        x2x6 = torch.ger(x2_pow, x6_pow).flatten()
        x3x4 = torch.ger(x3_pow, x4_pow).flatten()
        x3x5 = torch.ger(x3_pow, x5_pow).flatten()
        x3x6 = torch.ger(x3_pow, x6_pow).flatten()
        x4x5 = torch.ger(x4_pow, x4_pow).flatten()
        x4x6 = torch.ger(x4_pow, x6_pow).flatten()
        x5x6 = torch.ger(x5_pow, x6_pow).flatten()
        input = torch.cat((x1_pow, x2_pow, x3_pow, x4_pow, x5_pow, x6_pow,
                           x1x2, x1x3, x1x4, x1x4, x1x5, x1x6,
                           x2x3, x2x4, x2x5, x2x6,
                           x3x4, x3x5, x3x6,
                           x4x5, x4x6,
                           x5x6))
        out = self.fc1(input)
        return out

###############################################################################
# Used for Pendulum
###############################################################################

# Univariate trinomial. For use with a single point in 1D space (angular data).
class univariate_trinomial(nn.Module):
    def __init__(self):
        super(univariate_trinomial, self).__init__()
        self.fc1 = nn.Linear(3, 1, bias=False)

    def forward(self, x):
        x_power = torch.cat(3*[x]).squeeze(1)
        x_power = torch.pow(x_power, torch.Tensor([i for i in range(1, 4)]))
        out = self.fc1(x_power)
        return out

# Univariate trigonometric space. For use with a single point in 1D space (angular data).
class univariate_trigonometric_term(nn.Module):
    def __init__(self):
        super(univariate_trigonometric_term, self).__init__()
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.trig = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        x_sin = torch.sin(self.trig(x.squeeze(1)) + x.squeeze(1))
        out = self.fc1(x_sin)
        return out

class SSINN(nn.Module):
    def __init__(self, Tp, Vq, solver, tol=1e-3):
        super(SSINN, self).__init__()
        self.Tp = Tp
        self.Vq = Vq
        self.tol = tol
        self.solver = solver

    def forward(self, p0, q0, t0, t1):
        p, q = self.solver(p0, q0, t0, t1, self.Tp, self.Vq, self.tol)
        return p, q

def fourth_order(p0, q0, t0, t1, Tp, Vq, eps=0.1):
    h = t1 - t0
    kp = p0
    kq = q0
    c = torch.Tensor([0.5/(2.-2.**(1./3.)),
         (0.5-2.**(-2./3.))/(2.-2.**(1./3.)),
         (0.5-2.**(-2./3.))/(2.-2.**(1./3.)),
         0.5/(2.-2.**(1./3.))])
    d = torch.Tensor([1./(2.-2.**(1./3.)),
         -2.**(1./3.)/(2.-2.**(1./3.)),
         1./(2.-2.**(1./3.)),0.])
    for j in range(4):
        tp = kp
        tq = kq + c[j] * Tp(kp) * h
        kp = tp - d[j] * Vq(tq) * h
        kq = tq
    return kp, kq

def reward_nrmse(y_pred, y_rnn):
    """Compute NRMSE between predicted y and actual y
    """
    loss = nn.MSELoss()
    val = torch.sqrt(loss(y_pred, y_rnn)) # Convert to RMSE
    val = torch.std(y_rnn) * val # Normalize using stdev of targets
    val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10)) # Fix nan and clip
    val = 1 / (1 + val) # Squash
    return val.item()

if __name__=='__main__':
    main()
