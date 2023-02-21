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
    # Uncomment datasets amd feature_names accordingly
    with torch.no_grad():
        # Harmonic Oscillator with no constants
        # dataset = load_hamiltonian_dataset('.././datasets/harmonic_noconstant/harmonic_noconstant_noise_3.pkl')
        # feature_names=["q", "p"]
        # feature_library = ps.GeneralizedLibrary([ps.PolynomialLibrary()])

        # Harmonic Oscillator with constants
        # dataset = load_hamiltonian_dataset('.././datasets/pendulum/harmonic_noconstant_noise_3.pkl')
        # feature_names=["q_theta", "p_theta"]
        # feature_library = ps.GeneralizedLibrary([ps.PolynomialLibrary()])

        # Pendulum
        # dataset = load_hamiltonian_dataset('.././datasets/pendulum/pendulum_noise_3.pkl')
        # feature_names=["q_theta", "p_theta"]
        # feature_library = ps.GeneralizedLibrary([ps.FourierLibrary(), ps.PolynomialLibrary()])

        # Two-Body
        # dataset = load_hamiltonian_dataset('.././datasets/kepler/kepler_noise_1.pkl')
        # feature_names=["q_ix", "q_iy", "q_jx", "q_jy", "p_ix", "p_iy", "p_jx", "p_jy"]
        # feature_library = ps.GeneralizedLibrary([ps.PolynomialLibrary()])

        # Three-Body
        dataset = load_hamiltonian_dataset('.././datasets/3body/3body_noise_2.pkl')
        feature_names=["q_ix", "q_iy", "q_jx", "q_jy", "q_kx", "q_ky", "p_ix", "p_iy", "p_jx", "p_jy", "p_kx", "p_ky"]
        feature_library = ps.GeneralizedLibrary([ps.PolynomialLibrary()])

    dataset = np.array(dataset.detach().numpy()).astype('float64')
    t = np.linspace(0, len(dataset)*0.1-0.1, len(dataset))

    model = ps.SINDy(
        feature_names=feature_names,
        optimizer = ps.STLSQ(threshold=1.55),
        feature_library = feature_library
    )
    model.fit(dataset, t=t)
    model.print()

    predicted = [[dataset[0,j] for j in range(len(dataset[0]))]]
    for j in range(len(dataset)-1):
        sim = model.simulate([dataset[j,i] for i in range(len(dataset[0]))], t=[0,0.1])
        predicted.append(sim[1].tolist())
    print("Reward: ", round(reward_nrmse(torch.tensor(predicted), torch.tensor(dataset)), 3))

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
