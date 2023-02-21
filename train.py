###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# Original Paper: https://arxiv.org/abs/1912.04871 (Petersen et al)

# train.py: Contains main training loop (and reward functions) for PyTorch
# implementation of Deep Symbolic Regression.

###############################################################################
# Dependencies
###############################################################################

import time
import random
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from operators import Operators
from rnn import DSRRNN
from expressions import *
from collections import Counter

###############################################################################
# Main Training loop
###############################################################################

def train(
        X_constants,
        y_constants,
        X_rnn,
        y_rnn,
        mutation_rate = 0.0,
        operator_list = ['*', '+', '-', '/', '^', 'cos', 'sin', 'c', 'var_x', 'var_y'],
        left_variables = ['var_x'],
        right_variables = ['var_y'],
        custom_variables = {},
        left_min_length = 1,
        left_max_length = 6,
        right_min_length = 1,
        right_max_length = 6,
        left_separability = [],
        right_separability = [],
        left_symmetric = False,
        right_symmetric = False,
        root_node = '+',
        num_steps = 1,
        step_size = 0.1,
        integrator = 'symplectic',
        type = 'lstm',
        num_layers = 1,
        dropout = 0.0,
        lr = 0.0005,
        optimizer = 'adam',
        entropy_coefficient = 0.005,
        risk_factor = 0.95,
        initial_batch_size = 2000,
        scale_initial_risk = True,
        batch_size = 500,
        num_batches = 200,
        early_stopping_val = 1.0,
        hidden_size = 500,
        inner_optimizer = 'rmsprop',
        inner_lr = 0.1,
        inner_num_epochs = 15,
        use_gpu = False,
        live_print = True,
        summary_print = True,
    ):
    """Deep Symbolic Regression Training Loop

    ###
    Function Parameters
    ###
    ~ Training Data ~
    Note: There can be overlap (including complete) between constants/rnn data.
    - X_constants (Tensor): X dataset used for training constants
    - y_constants (Tensor): y dataset used for training constants
    - X_rnn (Tensor): X dataset used for obtaining reward / training RNN
    - y_rnn (Tensor): y dataset used for obtaining reward / training RNN

    ~ Expression Parameters ~
    - mutation_rate (float): between 0 and 1, specifies how frequently a random operator is sampled (within constraints)
    - operator_list (list of str): operators to use (all variables must have prefix var_)
    - left_variables (list of str): variables that can appear in left subtree (for separable)
    - right_variables (list of str): variables that can appear in right subtree (for separable)
    - left_min_length (int): minimum length of left subtree expression
    - left_max_length (int): maximum length of left subtree expression
    - right_min_length (int): minimum length of right subtree expression
    - right_max_length (int): maximum length of right subtree expression
    - left_separability (list of str): operator used to split left into terms
    - right_separability (list of str): operator used to split right into terms
    - root_node (str): operator used to define the root for separable expressions (generally '+')

    ~ Expression Reward Parameters ~
    - integrator (string): 'symplectic' (fourth-order) or 'rk4' (Runge-Kutta fourth-order)
    - num_steps (int): number of time steps to integrate before taking loss (reduces noise)
    - step_size (int): number of time steps between observations

    ~ Sequence-Generating Model Parameters ~
    - type ('rnn', 'lstm', or 'gru'): type of architecture to use
    - num_layers (int): number of layers in RNN architecture
    - dropout (float): dropout (if any) for RNN architecture
    - lr (float): learning rate for RNN
    - optimizer ('adam' or 'rmsprop'): optimizer for RNN
    - entropy_coefficient (float): entropy coefficient for RNN
    - risk_factor (float, >0, <1): we discard the bottom risk_factor quantile
    when training the RNN
    - batch_size (int): batch size for training the RNN
    - num_batches (int): number of batches (will stop early if found)
    - hidden_size (int): hidden dimension size for RNN

    ~ Expression Constant Optimization Parameters ~
    - inner_optimizer ('lbfgs', 'adam', or 'rmsprop'): optimizer for expressions
    - inner_lr (float): learning rate for constant optimization
    - inner_num_epochs (int): number of epochs for constant optimization

    ~ Misc. Training Details ~
    - use_gpu (bool): whethe or not to train with GPU (not currently working)
    - live_print (bool): if true, will print updates during training process
    - summary_print (bool): if true, will print summary of training process at termination

    ###
    Returns
    ###
    A list of four lists:
    [0] epoch_best_rewards (list of float): list of highest reward obtained each epoch
    [1] epoch_best_expressions (list of Expression): list of best expression each epoch
    [2] best_reward (float): best reward obtained
    [3] best_expression (Expression): best expression obtained
    """

    epoch_best_rewards = []
    epoch_best_expressions = []

    # Establish GPU device if necessary
    if (use_gpu and torch.cuda.is_available()):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Initialize operators, RNN, and optimizer
    operators = Operators(operator_list, device, left_variables, right_variables, custom_variables)
    dsr_rnn = DSRRNN(operators, hidden_size, device, left_separability, right_separability,
                     type=type, dropout=dropout, left_min_length=left_min_length,
                     left_max_length=left_max_length, right_min_length=right_min_length,
                     right_max_length=right_max_length, mutation_rate=mutation_rate,
                     ).to(device)
    if (optimizer == 'adam'):
        optim = torch.optim.Adam(dsr_rnn.parameters(), lr=lr)
    else:
        optim = torch.optim.RMSprop(dsr_rnn.parameters(), lr=lr)

    # Best expression and its performance
    best_expression, best_performance = None, float('-inf')

    # First sampling done outside of loop for initial batch size if desired
    start = time.time()
    sequences, sequence_lengths, log_probabilities, entropies = dsr_rnn.sample_separable_sequence(initial_batch_size, left_symmetric=left_symmetric, right_symmetric=right_symmetric)

    print(sequences)
    for i in range(num_batches):
        # Convert sequences into Pytorch expressions that can be evaluated
        expressions = []
        for j in range(len(sequences)):
            expressions.append(
                SymplecticExpression(
                    operators, sequences[j], sequence_lengths[j], integrator,
                    step_size, num_steps, left_separability, right_separability,
                    left_symmetric, right_symmetric
                ).to(device)
            )

        # Optimize constants of expressions (training data)
        optimize_constants(expressions, X_constants, y_constants, inner_lr, inner_num_epochs, inner_optimizer)

        # Benchmark expressions (test dataset)
        rewards = []
        for expression in expressions:
            rewards.append(benchmark(expression, X_rnn, y_rnn))
        rewards = torch.tensor(rewards)

        # Update best expression
        best_epoch_expression = expressions[np.argmax(rewards)]
        epoch_best_expressions.append(best_epoch_expression)
        epoch_best_rewards.append(max(rewards).item())
        if (max(rewards) > best_performance):
            best_performance = max(rewards)
            best_expression = best_epoch_expression

        # Early stopping criteria
        if (best_performance >= early_stopping_val):
            best_str = str(best_expression)
            if (live_print):
                print("~ Early Stopping Met ~")
                print(f"""Best Expression: {best_str}""")
            break

        # Compute risk threshold
        if (i == 0 and scale_initial_risk):
            threshold = np.quantile(rewards, 1 - (1 - risk_factor) / (initial_batch_size / batch_size))
        else:
            threshold = np.quantile(rewards, risk_factor)
        indices_to_keep = torch.tensor([j for j in range(len(rewards)) if rewards[j] > threshold])

        if (len(indices_to_keep) == 0 and summary_print):
            print("Threshold removes all expressions. Terminating.")
            break

        # Select corresponding subset of rewards, log_probabilities, and entropies
        rewards = torch.index_select(rewards, 0, indices_to_keep)
        log_probabilities = torch.index_select(log_probabilities, 0, indices_to_keep)
        entropies = torch.index_select(entropies, 0, indices_to_keep)

        # Compute risk seeking and entropy gradient
        risk_seeking_grad = torch.sum((rewards - threshold) * log_probabilities, axis=0)
        entropy_grad = torch.sum(entropies, axis=0)

        # Mean reduction and clip to limit exploding gradients
        risk_seeking_grad = torch.clip(risk_seeking_grad / len(rewards), -1e6, 1e6)
        entropy_grad = entropy_coefficient * torch.clip(entropy_grad / len(rewards), -1e6, 1e6)

        # Compute loss and backpropagate
        loss = -1 * lr * (risk_seeking_grad + entropy_grad)
        loss.backward()
        optim.step()

        # Epoch Summary
        if (live_print):
            print(f"""Epoch: {i+1} ({round(float(time.time() - start), 2)}s elapsed)
            Best Performance (Overall): {best_performance}
            Best Performance (Epoch): {max(rewards)}
            Best Expression (Epoch): {best_epoch_expression}
            """)
            # print(f"""Epoch: {i+1} ({round(float(time.time() - start), 2)}s elapsed)
            # Entropy Loss: {entropy_grad.item()}
            # Risk-Seeking Loss: {risk_seeking_grad.item()}
            # Total Loss: {loss.item()}
            # Best Performance (Overall): {best_performance}
            # Best Performance (Epoch): {max(rewards)}
            # Best Expression (Overall): {best_expression}
            # Best Expression (Epoch): {best_epoch_expression}
            # """)
        # Sample for next batch
        sequences, sequence_lengths, log_probabilities, entropies = dsr_rnn.sample_separable_sequence(batch_size, left_symmetric=left_symmetric, right_symmetric=right_symmetric)

    if (summary_print):
        print(f"""
        Time Elapsed: {round(float(time.time() - start), 2)}s
        Epochs Required: {i+1}
        Best Performance: {round(best_performance.item(),3)}
        Best Expression: {best_expression}
        """)

    return [epoch_best_rewards, epoch_best_expressions, best_performance, best_expression]

###############################################################################
# Reward function
###############################################################################

def benchmark(expression, X_rnn, y_rnn):
    """Obtain reward for a given expression using the passed X_rnn and y_rnn
    """
    y_pred = expression(X_rnn)
    return reward_nrmse(y_pred, y_rnn)

def reward_nrmse(y_pred, y_rnn):
    """Compute NRMSE between predicted y and actual y
    """
    loss = nn.MSELoss()
    val = torch.sqrt(loss(y_pred, y_rnn)) # Convert to RMSE
    val = torch.std(y_rnn) * val # Normalize using stdev of targets
    val = min(torch.nan_to_num(val, nan=1e10), torch.tensor(1e10)) # Fix nan and clip
    val = 1 / (1 + val) # Squash
    return val.item()

def reward_nrmse_grad(y_pred, y_rnn):
    """Computes NRMSE with grad tracked for supervised methods
    """
    loss = nn.MSELoss()
    val = torch.sqrt(loss(y_pred, y_rnn)) # Convert to RMSE
    val = torch.std(y_rnn) * val # Normalize using stdev of targets
    val = 1 / (1 + val) # Squash
    return val
