###############################################################################
# General Information
###############################################################################
# Data generation code for pendulum

# H(p, q) = p^2 / 2ml^2 + mgl(1 - cos(q))

###############################################################################
# Dependencies
###############################################################################

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as lines

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional
import numpy as np
import math
import random
import pickle

from expressions import GradientWrapper
from integrators import fourth_order, rk4

###############################################################################
# Main Data Generation Loop
###############################################################################

def main():
    # Dynamical System Hyperparameters
    # Dataset one
    m = 1.10 # Mass
    l = 0.73 # pendulum length
    g = 1.02 # gravity
    q0 = 1.37 # Initial angle (radians)
    p0 = 0.12 # Initial angular momentum

    # Simulation Hyperparameters
    n = 40 # Number of points
    dt = 0.1 # Time delta between each point
    std = 0.000 # St dev of Gaussian noise (0.0 if none desired)

    figure, axis = plt.subplots(2, 2)

    # Baseline
    _Tp = Tp(m, l)
    _Vq = Vq(m, l, g)
    Tp_expression = GradientWrapper(_Tp)
    Vq_expression = GradientWrapper(_Vq)
    baseline_data = get_data(dt, q0, p0, n, Tp_expression, Vq_expression)
    x_plot_b = baseline_data[:, 0, 0].tolist()
    y_plot_b = baseline_data[:, 0, 1].tolist()

    _Tp = Tp_2(m, l)
    _Vq = Vq_2(m, l, g)
    Tp_expression = GradientWrapper(_Tp)
    Vq_expression = GradientWrapper(_Vq)
    data = get_data(dt, q0, p0, n, Tp_expression, Vq_expression)
    x_plot = data[:, 0, 0].tolist()
    y_plot = data[:, 0, 1].tolist()
    axis[0, 0].plot(x_plot_b, y_plot_b, '--')
    axis[0, 0].plot(x_plot, y_plot)
    axis[0, 0].set_xlim([-2.5, 2])
    axis[0, 0].set_ylim([-1.5, 1.5])
    axis[0, 0].set_title("Epoch 1")

    _Tp = Tp_9(m, l)
    _Vq = Vq_9(m, l, g)
    Tp_expression = GradientWrapper(_Tp)
    Vq_expression = GradientWrapper(_Vq)
    data = get_data(dt, q0, p0, n, Tp_expression, Vq_expression)
    x_plot = data[:, 0, 0].tolist()
    y_plot = data[:, 0, 1].tolist()
    axis[0, 1].plot(x_plot_b, y_plot_b, '--')
    axis[0, 1].plot(x_plot, y_plot)
    axis[0, 1].set_xlim([-2.5, 2])
    axis[0, 1].set_ylim([-1.5, 1.5])
    axis[0, 1].set_title("Epoch 9")

    _Tp = Tp_14(m, l)
    _Vq = Vq_14(m, l, g)
    Tp_expression = GradientWrapper(_Tp)
    Vq_expression = GradientWrapper(_Vq)
    data = get_data(dt, q0, p0, n, Tp_expression, Vq_expression)
    x_plot = data[:, 0, 0].tolist()
    y_plot = data[:, 0, 1].tolist()
    axis[1, 1].plot(x_plot_b, y_plot_b, '--')
    axis[1, 1].plot(x_plot, y_plot)
    axis[1, 1].set_xlim([-2.5, 2])
    axis[1, 1].set_ylim([-1.5, 1.5])
    axis[1, 1].set_title("Epoch 14")

    _Tp = Tp_28(m, l)
    _Vq = Vq_28(m, l, g)
    Tp_expression = GradientWrapper(_Tp)
    Vq_expression = GradientWrapper(_Vq)
    data = get_data(dt, q0, p0, n, Tp_expression, Vq_expression)
    x_plot = data[:, 0, 0].tolist()
    y_plot = data[:, 0, 1].tolist()
    axis[1, 0].plot(x_plot_b, y_plot_b, '--')
    axis[1, 0].plot(x_plot, y_plot)
    axis[1, 0].set_xlim([-2.5, 2])
    axis[1, 0].set_ylim([-1.5, 1.5])
    axis[1, 0].set_title("Epoch 28")

    # Combine all the operations and display
    plt.show()


###############################################################################
# Dynamical System Expressions
###############################################################################

class Tp(nn.Module):
    def __init__(self, m, l):
        super(Tp, self).__init__()
        self.m = m
        self.l = l
    def forward(self, p):
        return p**2 / (2 * self.m * self.l**2)

class Vq(nn.Module):
    def __init__(self, m, l, g):
        super(Vq, self).__init__()
        self.m = m
        self.l = l
        self.g = g
    def forward(self, q):
        return self.m * self.g * self.l * (1 - torch.cos(q))

### Epoch 1
class Tp_1(nn.Module):
    def __init__(self, m, l):
        super(Tp_1, self).__init__()
        self.m = m
        self.l = l
    def forward(self, p):
        return torch.sin(((p + -0.9804) * p) + p)
class Vq_1(nn.Module):
    def __init__(self, m, l, g):
        super(Vq_1, self).__init__()
        self.m = m
        self.l = l
        self.g = g
    def forward(self, q):
        return (q * (q * 0.3164)) + 0.5306

### Epoch 2
class Tp_2(nn.Module):
    def __init__(self, m, l):
        super(Tp_2, self).__init__()
        self.m = m
        self.l = l
    def forward(self, p):
        return ((1.2545 - 0.3933) * p) * p
class Vq_2(nn.Module):
    def __init__(self, m, l, g):
        super(Vq_2, self).__init__()
        self.m = m
        self.l = l
        self.g = g
    def forward(self, q):
        return torch.sin(((-4.9387 / 5.3273) / 5.4833) * q)

### Epoch 5
class Tp_5(nn.Module):
    def __init__(self, m, l):
        super(Tp_5, self).__init__()
        self.m = m
        self.l = l
    def forward(self, p):
        return p * ((0.5151 + p) + -0.3706)
class Vq_5(nn.Module):
    def __init__(self, m, l, g):
        super(Vq_5, self).__init__()
        self.m = m
        self.l = l
        self.g = g
    def forward(self, q):
        return q-q

### Epoch 9
class Tp_9(nn.Module):
    def __init__(self, m, l):
        super(Tp_9, self).__init__()
        self.m = m
        self.l = l
    def forward(self, p):
        return ((p + 0.0354) + -0.0278) * p
class Vq_9(nn.Module):
    def __init__(self, m, l, g):
        super(Vq_9, self).__init__()
        self.m = m
        self.l = l
        self.g = g
    def forward(self, q):
        return torch.cos(((q * 3.5897) / q) + q)

### Epoch 14
class Tp_14(nn.Module):
    def __init__(self, m, l):
        super(Tp_14, self).__init__()
        self.m = m
        self.l = l
    def forward(self, p):
        return ((p * 0.8519) * p) + 0.8478
class Vq_14(nn.Module):
    def __init__(self, m, l, g):
        super(Vq_14, self).__init__()
        self.m = m
        self.l = l
        self.g = g
    def forward(self, q):
        return torch.cos(((3.0457 - q) - q) + q)

### Epoch 28
class Tp_28(nn.Module):
    def __init__(self, m, l):
        super(Tp_28, self).__init__()
        self.m = m
        self.l = l
    def forward(self, p):
        return ((p + p) * p) * 0.4265
class Vq_28(nn.Module):
    def __init__(self, m, l, g):
        super(Vq_28, self).__init__()
        self.m = m
        self.l = l
        self.g = g
    def forward(self, q):
        return 0.819 * torch.sin(-1.5712 + q)



def get_data(dt, q0, p0, n, Tp_expression, Vq_expression):
    t0 = torch.Tensor([[0.]])
    t1 = torch.Tensor([[dt]])
    q0 = torch.Tensor([[q0]])
    p0 = torch.Tensor([[p0]])
    data = [torch.cat((q0, p0), axis=1)]
    for i in range(n):
        p0 = Variable(p0, requires_grad=True)
        q0 = Variable(q0, requires_grad=True)
        p1, q1 = fourth_order(p0, q0, t0, t1, Tp_expression, Vq_expression)
        data.append(torch.cat((q1, p1), axis=1))
        p0, q0 = p1, q1
    # ~ Stack into a single tensor ~
    data = torch.stack(data, axis=0)
    return data

if __name__ == '__main__':
    main()
