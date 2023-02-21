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
    m = 0.47 # Mass
    l = 1.23 # pendulum length
    g = 1.95 # gravity
    q0 = 1.32 # Initial angle (radians)
    p0 = 0.23 # Initial angular momentum

    # Dataset two
    # m = 1.10 # Mass
    # l = 0.73 # pendulum length
    # g = 1.02 # gravity
    # q0 = 1.37 # Initial angle (radians)
    # p0 = 0.12 # Initial angular momentum

    # # Dataset three
    # m = 0.68 # Mass
    # l = 0.33 # pendulum length
    # g = 1.59 # gravity
    # q0 = 0.87 # Initial angle (radians)
    # p0 = 0.15 # Initial angular momentum

    # Simulation Hyperparameters
    n = 50 # Number of points
    dt = 0.1 # Time delta between each point
    std = 0.001 # St dev of Gaussian noise (0.0 if none desired)

    _Tp = Tp(m, l)
    _Vq = Vq(m, l, g)
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
        'l': l,
        'g': g,
        'q0': q0,
        'p0': p0,
        'n': n,
        'dt': dt,
        'std': std,
        'data': data
    }

    # Save data
    file_name = f"""pendulum_m{m}_l{l}_g{g}_std{std}"""
    with open("../datasets/" + file_name + ".pkl", 'wb') as handle:
        pickle.dump(data_dictionary, handle)
        print('Dataset saved in ../datasets/{}.pkl'.format(file_name))

    # Plot animation of motion
    plot_data = data[:, 0, 0].tolist()
    xs = [l * math.cos(x+4.71) for x in plot_data]
    ys = [l * math.sin(x+4.71) for x in plot_data]
    step = [i*dt for i in range(0, n)]
    fig, ax = plt.subplots()
    mat, = ax.plot(0, 0, 'o')
    ax.axis([min(xs+ys)-0.05, max(xs+ys)+0.05,min(xs+ys)-0.05, max(xs+ys)+0.05])
    ax.set_title("Pendulum Motion")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel('x')
    def harmonic_animation(i):
        if (len(ax.lines) > 0):
            ax.lines.remove(ax.lines[0])
        line = lines.Line2D([0, xs[i]],
                    [0,ys[i]],
                    lw = 1, color ='blue',
                    solid_capstyle = 'round', marker='o',
                    axes = ax, alpha = 0.7)
        ax.add_line(line)
        mat.set_data(xs[i], ys[i])
        ax.text(0.5, 0.5, 'Time: {}'.format(round(step[i], 2)), backgroundcolor='w')
        return mat,
    ani = animation.FuncAnimation(fig, harmonic_animation, interval=100, frames=range(1, len(step)), repeat=True)
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

if __name__ == '__main__':
    main()
