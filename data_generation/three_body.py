###############################################################################
# General Information
###############################################################################
# Data generation code for coupled oscillator

# H(p,q) = (0.5) * (px^2 + py^2 + qx^2 + qy^2) + qy * qx^2 - (1/3) * qy^3

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
    # Dataset one
    q0i_x = 0.00 # x position coordinate of body i
    q0i_y = 0.00 # y position coordinate of body i
    q0j_x = 1.00 # x position coordinate of body j
    q0j_y = 1.00 # y position coordinate of body j
    q0k_x = -1.00 # x position coordinate of body k
    q0k_y = -1.00 # y position coordinate of body k
    p0i_x = 0.50 # x momentum coordinate of body i
    p0i_y = 0.30 # y momentum coordinate of body i
    p0j_x = 1.10 # x momentum coordinate of body j
    p0j_y = -0.40 # y momentum coordinate of body j
    p0k_x = -0.20 # x momentum coordinate of body k
    p0k_y = 0.50 # y momentum coordinate of body k

    # Dataset two
    # q0i_x = 0.00 # x position coordinate of body i
    # q0i_y = 0.00 # y position coordinate of body i
    # q0j_x = -3.00 # x position coordinate of body j
    # q0j_y = 0.00 # y position coordinate of body j
    # q0k_x = 3.00 # x position coordinate of body k
    # q0k_y = 0.00 # y position coordinate of body k
    # p0i_x = 0.00 # x momentum coordinate of body i
    # p0i_y = -1.00 # y momentum coordinate of body i
    # p0j_x = 0.50 # x momentum coordinate of body j
    # p0j_y = 0.00 # y momentum coordinate of body j
    # p0k_x = 0.50 # x momentum coordinate of body k
    # p0k_y = 0.00 # y momentum coordinate of body k

    # Dataset three
    # q0i_x = 0.25 # x position coordinate of body i
    # q0i_y = 0.25 # y position coordinate of body i
    # q0j_x = 2.00 # x position coordinate of body j
    # q0j_y = 1.50 # y position coordinate of body j
    # q0k_x = 3.00 # x position coordinate of body k
    # q0k_y = -2.00 # y position coordinate of body k
    # p0i_x = 0.00 # x momentum coordinate of body i
    # p0i_y = -1.00 # y momentum coordinate of body i
    # p0j_x = 0.50 # x momentum coordinate of body j
    # p0j_y = -0.15 # y momentum coordinate of body j
    # p0k_x = 0.80 # x momentum coordinate of body k
    # p0k_y = 0.25 # y momentum coordinate of body k

    m_i = 1 # mass of body i
    m_j = 1 # mass of body j
    m_k = 1 # mass of body k

    # Simulation Hyperparameters
    n = 250 # Number of points
    dt = 0.1 # Time delta between each point
    std = 0.001 # St dev of Gaussian noise (0.0 if none desired)

    _Tp = Tp(m_i, m_j, m_k)
    _Vq = Vq(m_i, m_j, m_k)
    Tp_expression = GradientWrapper(_Tp)
    Vq_expression = GradientWrapper(_Vq)

    t0 = torch.Tensor([[0.]])
    t1 = torch.Tensor([[dt]])
    q0 = torch.Tensor([[q0i_x, q0i_y, q0j_x, q0j_y, q0k_x, q0k_y]])
    p0 = torch.Tensor([[p0i_x, p0i_y, p0j_x, p0j_y, p0k_x, p0k_y]])
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
            noise = torch.empty(1,12).normal_(mean=0,std=std)
            data[i] += noise

    # ~ Stack into a single tensor ~
    data = torch.stack(data, axis=0)
    print(data)
    # Assemble data dictionary
    data_dictionary = {
        'q0i_x': q0i_x,
        'q0i_y': q0i_y,
        'q0j_x': q0j_x,
        'q0j_y': q0j_y,
        'q0k_x': q0k_x,
        'q0k_y': q0k_y,
        'p0i_x': p0i_x,
        'p0i_y': p0i_y,
        'p0j_x': p0j_x,
        'p0j_y': p0j_y,
        'p0k_x': p0k_x,
        'p0k_y': p0k_y,
        'm_i': m_i,
        'm_j': m_j,
        'm_k': m_k,
        'n': n,
        'dt': dt,
        'std': std,
        'data': data
    }

    # Save data
    file_name = f"""3body_mi_{m_i}_mj_{m_j}_mk_{m_k}_std{std}"""
    with open("../datasets/" + file_name + ".pkl", 'wb') as handle:
        pickle.dump(data_dictionary, handle)
        print('Dataset saved in ../datasets/{}.pkl'.format(file_name))

    # Plot animation of motion
    x_i_data = data[:, 0, 0].tolist()
    y_i_data = data[:, 0, 1].tolist()
    x_j_data = data[:, 0, 2].tolist()
    y_j_data = data[:, 0, 3].tolist()
    x_k_data = data[:, 0, 4].tolist()
    y_k_data = data[:, 0, 5].tolist()

    min_x = min(x_i_data + x_j_data + x_k_data)
    max_x = max(x_i_data + x_j_data + x_k_data)
    min_y = min(y_i_data + y_j_data + y_k_data)
    max_y = max(y_i_data + y_j_data + y_k_data)

    step = [i*dt for i in range(0, n)]
    fig, ax = plt.subplots()
    mat, = ax.plot(0, 0, 'o')
    ax.axis([min_x-0.05, max_x+0.05,min_y-0.05, max_y+0.05])
    ax.set_title("Three Body Motion")
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel('x')
    def harmonic_animation(i):
        mat.set_data(
            [x_i_data[i], x_j_data[i], x_k_data[i]],
            [y_i_data[i], y_j_data[i], y_k_data[i]]
        )
        # ax.text(0, 0.05, 'Time: {}'.format(round(step[i], 2)), backgroundcolor='w')
        return mat,
    ani = animation.FuncAnimation(fig, harmonic_animation, interval=10, frames=range(1, len(step)), repeat=True)
    plt.show()

###############################################################################
# Dynamical System Expressions
###############################################################################

class Tp(nn.Module):
    def __init__(self, m_i, m_j, m_k):
        self.m_i = m_i
        self.m_j = m_j
        self.m_k = m_k
        super(Tp, self).__init__()
    def forward(self, p):
        p_i_x = p[:, 0]
        p_i_y = p[:, 1]
        p_j_x = p[:, 2]
        p_j_y = p[:, 3]
        p_k_x = p[:, 4]
        p_k_y = p[:, 5]
        Tp_i = (p_i_x**2 / 2 * self.m_i) + (p_i_y**2 / 2 * self.m_i)
        Tp_j = (p_j_x**2 / 2 * self.m_j) + (p_j_y**2 / 2 * self.m_j)
        Tp_k = (p_k_x**2 / 2 * self.m_k) + (p_k_y**2 / 2 * self.m_k)
        return Tp_i + Tp_j + Tp_k

class Vq(nn.Module):
    def __init__(self, m_i, m_j, m_k):
        self.m_i = m_i
        self.m_j = m_j
        self.m_k = m_k
        super(Vq, self).__init__()
    def forward(self, q):
        q_i_x = q[:, 0]
        q_i_y = q[:, 1]
        q_j_x = q[:, 2]
        q_j_y = q[:, 3]
        q_k_x = q[:, 4]
        q_k_y = q[:, 5]
        i_j_term = -1 * (self.m_i * self.m_j) / torch.sqrt( (q_j_x - q_i_x)**2 + (q_j_y - q_i_y)**2 )
        i_k_term = -1 * (self.m_i * self.m_k) / torch.sqrt( (q_k_x - q_i_x)**2 + (q_k_y - q_i_y)**2 )
        j_k_term = -1 * (self.m_j * self.m_k) / torch.sqrt( (q_k_x - q_j_x)**2 + (q_k_y - q_j_y)**2 )
        return i_j_term + i_k_term + j_k_term

if __name__ == '__main__':
    main()
