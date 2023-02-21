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

# def main():
#     # Load training and test data
#     datasets = ['datasets/pendulum/pendulum_nonoise_1.pkl', 'datasets/pendulum/pendulum_nonoise_2.pkl', 'datasets/pendulum/pendulum_nonoise_3.pkl']
#     datasets += ['datasets/pendulum/pendulum_noise_1.pkl', 'datasets/pendulum/pendulum_noise_2.pkl', 'datasets/pendulum/pendulum_noise_3.pkl']
#     for data in datasets:
#         print(data)
#         with torch.no_grad():
#             dataset = load_hamiltonian_dataset(data)
#             X, y = separate_xy(dataset, num_steps=1)
#
#         # Perform the regression task
#         results = train(
#             X,
#             y,
#             X,
#             y,
#             mutation_rate = 0.00,
#             operator_list = ['+', '*', '/', '-', 'c', 'cos', 'sin', 'var_q', 'var_p'],
#             left_variables = [['var_q']],
#             right_variables = [['var_p']],
#             custom_variables = {},
#             left_min_length = [1],
#             left_max_length = [8],
#             right_min_length = [1],
#             right_max_length = [8],
#             left_separability = [],
#             right_separability = [],
#             left_symmetric = False,
#             right_symmetric = False,
#             num_steps = 1,
#             step_size = 0.1,
#             integrator = 'rk4',
#             type = 'rnn',
#             num_layers = 1,
#             hidden_size = 250,
#             dropout = 0.0,
#             lr = 0.0005,
#             optimizer = 'adam',
#             inner_optimizer = 'rmsprop',
#             inner_lr = 0.5,
#             inner_num_epochs = 20,
#             entropy_coefficient = 0.005,
#             risk_factor = 0.95,
#             initial_batch_size = 500,
#             scale_initial_risk = True,
#             batch_size = 500,
#             num_batches = 30,
#             use_gpu = False,
#             live_print = False,
#             summary_print = True
#         )
#
#     # Unpack results
#     epoch_best_rewards = results[0]
#     epoch_best_expressions = results[1]
#     best_reward = results[2]
#     best_expression = results[3]
#
#     # Plot best rewards each epoch
#     plt.plot([i+1 for i in range(len(epoch_best_rewards))], epoch_best_rewards)
#     plt.xlabel('Epoch')
#     plt.ylabel('Reward')
#     plt.title('Reward over Time')
#     plt.show()

def main():
    # Load training and test data
    datasets = ['datasets/3body/3body_noise_1.pkl', 'datasets/3body/3body_noise_2.pkl', 'datasets/3body/3body_noise_3.pkl']
    for data in datasets:
        print(data)
        with torch.no_grad():
            dataset = load_hamiltonian_dataset(data)
            X, y = separate_xy(dataset, num_steps=10)
        # Coupling
        # detect_coupling(
        #     X, y,
        #     left_vars=['var_qix', 'var_qiy', 'var_qjx', 'var_qjy', 'var_qjx', 'var_qjy'],
        #     right_vars=['var_pix', 'var_piy', 'var_pjx', 'var_pjy', 'var_pjx', 'var_pjy']
        # )
        for i in range(5):
            print(i)
            # Perform the regression task
            results = train(
                X,
                y,
                X,
                y,
                mutation_rate = 0.05,
                operator_list = ['+', '*', '/', '-', 'c', 'var_pix', 'var_piy', 'var_pjx', 'var_pjy' ,'var_pkx', 'var_pky', 'var_euclidean1',  'var_euclidean2',  'var_euclidean3'],
                left_variables = [['var_euclidean1'], ['var_euclidean2'], ['var_euclidean3']],
                right_variables = [['var_pix'], ['var_piy'], ['var_pjx'], ['var_pjy'], ['var_pkx'], ['var_pky']],
                custom_variables = {
                    'var_euclidean1': 'torch.sqrt(torch.square(x[:, 2] - x[:, 0]) + torch.square(x[:, 3] - x[:, 1]))',
                    'var_euclidean2': 'torch.sqrt(torch.square(x[:, 4] - x[:, 2]) + torch.square(x[:, 5] - x[:, 3]))',
                    'var_euclidean3': 'torch.sqrt(torch.square(x[:, 4] - x[:, 0]) + torch.square(x[:, 5] - x[:, 1]))',
                },
                left_min_length = [1, 1, 1],
                left_max_length = [4, 4, 4],
                right_min_length = [1, 1, 1, 1, 1, 1],
                right_max_length = [6, 6, 6, 6, 6, 6],
                left_separability = ['+', '+'],
                right_separability = ['+', '+', '+', '+', '+'],
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
                initial_batch_size = 2000,
                scale_initial_risk = True,
                batch_size = 500,
                num_batches = 10,
                use_gpu = False,
                live_print = True,
                summary_print = True
            )

if __name__=='__main__':
    main()
