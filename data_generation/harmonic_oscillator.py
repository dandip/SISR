###############################################################################
# General Information
###############################################################################
# Data generation code for simple harmonic motion

# H(p,q) = (p^2 / 2m) + 0.5 * m * w^2 * q^2 where w is the classical oscillator
# frequency defined as w = sqrt(k / m).
# V(q) = 0.5 * m * w^2 * q^2
# T(p) = (p^2 / 2m)
# Link: https://quantummechanics.ucsd.edu/ph130a/130_notes/node153.html

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
    # (No constants required)
    m = 0.5 # Mass
    w = 2 # Classical oscillator frequency

    # Dataset one
    # q0 = -0.05
    # p0 = 0.42
    # m = 1.23
    # w = 1.65

    # Dataset two
    # p0 = -0.29
    # q0 = 0.27
    # m = 0.63
    # w = 1.30

    # Dataset three
    q0 = 0.06
    p0 = -0.54
    m = 1.69
    w = 0.83

    # Simulation Hyperparameters
    n = 30 # Number of points
    dt = 0.1 # Time delta between each point
    std = 0.001 # St dev of Gaussian noise (0.0 if none desired)

    _Tp = Tp(m)
    _Vq = Vq(m, w)
    Tp_expression = GradientWrapper(_Tp)
    Vq_expression = GradientWrapper(_Vq)

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

    # Add noise if desired
    if (std > 0):
        for i in range(len(data)):
            noise = torch.empty(1,2).normal_(mean=0,std=std)
            data[i] += noise

    # ~ Stack into a single tensor ~
    data = torch.stack(data, axis=0)

    # Assemble data dictionary
    data_dictionary = {
        'm': m,
        'w': w,
        'q0': q0,
        'p0': p0,
        'n': n,
        'dt': dt,
        'std': std,
        'data': data
    }

    # Save data
    file_name = f"""harmonic_oscillator_m{m}_w{w}_std{std}"""
    with open("../datasets/" + file_name + ".pkl", 'wb') as handle:
        pickle.dump(data_dictionary, handle)
        print('Dataset saved in ../datasets/{}.pkl'.format(file_name))

    # Plot animation of motion
    plot_data = data[:, 0, 0].tolist()
    step = [i*dt for i in range(0, n)]
    for i in range(20): # Add the last point a bit so that there's delay before repeat
        plot_data.append(plot_data[-1])
        step.append(step[-1])
    fig, ax = plt.subplots()
    mat, = ax.plot(0, 0, 'o')
    ax.axis([min(plot_data)-0.05, max(plot_data)+0.05,-0.1,0.1])
    ax.set_title("1D Simple Harmonic Oscillator Motion")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel('x')
    def harmonic_animation(i):
        mat.set_data(plot_data[i], 0)
        ax.text(0, 0.05, 'Time: {}'.format(round(step[i], 2)), backgroundcolor='w')
        return mat,
    ani = animation.FuncAnimation(fig, harmonic_animation, interval=100, frames=range(1, len(step)), repeat=True)
    plt.show()

###############################################################################
# Dynamical System Expressions
###############################################################################

class Tp(nn.Module):
    def __init__(self, m):
        super(Tp, self).__init__()
        self.m = m
    def forward(self, p):
        return p**2 / (2 * self.m)

class Vq(nn.Module):
    def __init__(self, m, w):
        super(Vq, self).__init__()
        self.m = m
        self.w = w
    def forward(self, q):
        return 0.5 * self.m * self.w**2 * q**2

if __name__ == '__main__':
    main()
