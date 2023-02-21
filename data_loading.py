###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip
# data_loading.py:

###############################################################################
# Dependencies
###############################################################################

import pickle
import numpy as np
import torch

###############################################################################
# Data Loading Functions
###############################################################################

def load_hamiltonian_dataset(dataset_str):
    """Loads serialized pickle dataset with attribute 'data' containing a time
    series of q, p observations
    """
    with open(dataset_str, 'rb') as f:
        pickle_obj = pickle.load(f)
    return pickle_obj['data'][:, 0, :]

###############################################################################
# Default Datasets
###############################################################################

def example_dataset_one():
    """Constructs data for Y = 2X0 + X1
    """
    # y = x[0] + 2*x[1]
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 0],
        [0, 2],
        [2, 2],
        [3, 0],
        [0, 3],
        [3, 3],
    ])
    y = 2*X[:, 0] + X[:, 1]
    return X, y

###############################################################################
# Data Manipulation Functions
###############################################################################

def separate_xy(dataset, num_steps=1):
    """Given time series of Hamiltonian data, separates into (t_{i}, t_{i+n})
    pairs that can be used for training where n is num_steps
    """
    X_indices = torch.Tensor([i for i in range(0, len(dataset) - num_steps)]).long()
    y_indices = torch.Tensor([i for i in range(num_steps, len(dataset))]).long()
    X = torch.index_select(dataset, 0, X_indices)
    y = torch.index_select(dataset, 0, y_indices)
    return X, y

def split():
    # Split randomly
    comb = list(zip(X, y))
    random.shuffle(comb)
    X, y = zip(*comb)

    # Proportion used to train constants versus benchmarking functions
    training_proportion = 0.2
    div = int(training_proportion*len(X))
    X_constants, X_rnn = np.array(X[:div]), np.array(X[div:])
    y_constants, y_rnn = np.array(y[:div]), np.array(y[div:])
    X_constants, X_rnn = torch.Tensor(X_constants), torch.Tensor(X_rnn)
    y_constants, y_rnn = torch.Tensor(y_constants), torch.Tensor(y_rnn)
    return X_constants, X_rnn, y_constants, y_rnn
