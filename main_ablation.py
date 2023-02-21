###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# Original Paper: https://arxiv.org/abs/1912.04871 (Petersen et al)

# main.py: From here, launch deep symbolic regression tasks. All
# hyperparameters are exposed (info on them can be found in train.py). Unless
# you'd like to impose new constraints / make significant modifications,
# modifying this file (and specifically the get_data function) is likely all
# you need to do for a new symbolic regression task.

###############################################################################
# Dependencies
###############################################################################

from data_loading import *
from coupling_detection import *
from train import train
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

###############################################################################
# Main Function
###############################################################################

# A note on operators:
# Available operators are: '*', '+', '-', '/', '^', 'sin', 'cos', 'tan',
#   'sqrt', 'square', and 'c.' You may also include constant floats, but they
#   must be strings. For variable operators, you must use the prefix var_.
#   Variable should be passed in the order that they appear in your data, i.e.
#   if your input data is structued [[x1, y1] ... [[xn, yn]] with outputs
#   [z1 ... zn], then var_x should precede var_y.

def main():
    # Load training and test data
    datasets = ['datasets/harmonic_constant/harmonic_constant_noise_1.pkl', 'datasets/harmonic_constant/harmonic_constant_noise_2.pkl', 'datasets/harmonic_constant/harmonic_constant_noise_3.pkl']
    # datasets = ['datasets/kepler/kepler_noise_1.pkl', 'datasets/kepler/kepler_noise_2.pkl', 'datasets/kepler/kepler_noise_3.pkl']
    # datasets = ['datasets/3body/3body_noise_1.pkl', 'datasets/3body/3body_noise_2.pkl', 'datasets/3body/3body_noise_3.pkl']
    for data in datasets:
        print(data)
        with torch.no_grad():
            dataset = load_hamiltonian_dataset(data)
            X, y = separate_xy(dataset, num_steps=1)

        # Perform the regression task
        results = train(
            X,
            y,
            X,
            y,
            mutation_rate = 0.05,
            operator_list = ['+', '*', '/', '-', 'c', 'cos', 'sin', 'var_q', 'var_p'],
            left_variables = [['var_q']],
            right_variables = [['var_p']],
            custom_variables = {
            },
            left_min_length = [1],
            left_max_length = [8],
            right_min_length = [1],
            right_max_length = [8],
            left_separability = [],
            right_separability = [],
            left_symmetric = True,
            right_symmetric = True,
            num_steps = 1,
            step_size = 0.1,
            integrator = 'symplectic',
            type = 'lstm',
            num_layers = 2,
            hidden_size = 250,
            dropout = 0.0,
            lr = 0.0005,
            optimizer = 'adam',
            inner_optimizer = 'rmsprop',
            inner_lr = 0.5,
            inner_num_epochs = 20,
            entropy_coefficient = 0.005,
            risk_factor = 0.95,
            initial_batch_size = 500,
            # initial_batch_size = 2000,
            scale_initial_risk = True,
            batch_size = 500,
            num_batches = 5,
            use_gpu = False,
            live_print = True,
            summary_print = True
        )

if __name__=='__main__':
    main()
