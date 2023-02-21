###############################################################################
# General Information
###############################################################################
# Author: Daniel DiPietro | dandipietro.com | https://github.com/dandip

# Original Paper: https://arxiv.org/abs/1912.04871 (Petersen et al)

# expression.py: Contains code for converting sampled sequences to
# Pytorch expressions. These expressions can then have their reward assessed,
# constants optimized, etc

###############################################################################
# Dependencies
###############################################################################

import time
from collections import Counter

import torch
import torch.nn as nn
from torch.autograd import Variable

from integrators import fourth_order, rk4

import warnings
warnings.filterwarnings("ignore")

###############################################################################
# Converting Sequence to Operators
###############################################################################

class OperatorNode:
    def __init__(self, operator, operators, arity, parent=None):
        """Description here
        """
        self.operator = operator
        self.operator_str = operators.operator_list[operator]
        self.arity = arity
        self.parent = parent
        self.left_child = None
        self.right_child = None

    def add_child(self, node):
        if (self.left_child is None):
            self.left_child = node
        elif (self.right_child is None):
            self.right_child = node
        else:
            raise RuntimeError("Both children have been created.")

    def set_parent(self, node):
        self.parent = node

    def remaining_children(self):
        if (self.arity == 0):
            return False
        elif (self.arity == 1 and self.left_child is not None):
            return False
        elif (self.arity == 2 and self.left_child is not None and self.right_child is not None):
            return False
        return True

    def __str__(self):
        return str(self.operator)

    def recursive_print(self, symmetric):
        print_val = self.recursive_print_helper()
        num_constants = print_val.count('@') # Placeholder for constants
        for i in range(num_constants):
            if (not symmetric):
                print_val = print_val.replace('@', 'c' + str(i), 1)
            else:
                print_val = print_val.replace('@', 'c' + str(0), 1)
        return print_val

    def recursive_print_helper(self):
        if (self.arity == 2):
            left_print = self.left_child.recursive_print_helper()
            right_print = self.right_child.recursive_print_helper()
            if (self.left_child.arity == 2):
                left_print = '(' + left_print + ')'
            if (self.right_child.arity == 2):
                right_print = '(' + right_print + ')'
            return str(f"""{left_print} {self.operator_str} {right_print}""")
        elif (self.arity == 1):
            return str(f"""{self.operator_str}({self.left_child.recursive_print_helper()})""")
        else:
            if (self.operator_str == 'c'):
                return str('@')
            elif ('var_' in self.operator_str):
                return str(self.operator_str.strip('var_'))
            else:
                return str(self.operator_str)

    def torch_print(self, operators):
        torch_val = self.torch_print_helper(operators)
        num_constants = torch_val.count('@') # Placeholder for constants
        for i in range(num_constants):
            torch_val = torch_val.replace('@', 'self.c[' + str(i) + ']', 1)
        return torch_val

    def torch_print_helper(self, operators):
        if (self.arity == 2):
            left_print = self.left_child.torch_print_helper(operators)
            right_print = self.right_child.torch_print_helper(operators)
            return str(f"""{operators.func_i(self.operator)}({left_print}, {right_print})""")
        elif (self.arity == 1):
            return str(f"""{operators.func_i(self.operator)}({self.left_child.torch_print_helper(operators)})""")
        else:
            if (operators.operator_list[self.operator] == 'c'):
                return '@'
            elif ('var_' in operators.operator_list[self.operator]):
                return str('x[:,' + str(operators.var_i(self.operator)) + ']')
            else:
                return 'torch.tensor([' + str(self.operator_str) + '])'

    def torch_child_print(self, operators, side, symmetric):
        torch_val = self.torch_child_print_helper(operators, side)
        num_constants = torch_val.count('@') # Placeholder for constants
        for i in range(num_constants):
            if (not symmetric):
                torch_val = torch_val.replace('@', 'self.c[' + str(i) + ']', 1)
            else:
                torch_val = torch_val.replace('@', 'self.c[' + str(0) + ']', 1)
        if (not symmetric):
            return torch_val, num_constants
        else:
            return torch_val, min(num_constants, 1)

    def torch_child_print_helper(self, operators, side):
        if (self.arity == 2):
            left_print = self.left_child.torch_child_print_helper(operators, side)
            right_print = self.right_child.torch_child_print_helper(operators, side)
            return str(f"""{operators.func_i(self.operator)}({left_print}, {right_print})""")
        elif (self.arity == 1):
            return str(f"""{operators.func_i(self.operator)}({self.left_child.torch_child_print_helper(operators, side)})""")
        else:
            if (operators.operator_list[self.operator] == 'c'):
                return '@'
            elif ('var_' in operators.operator_list[self.operator]):
                return str(operators.var_i_side_tensor(self.operator, side))
            else:
                return 'torch.tensor([' + str(self.operator_str) + '])'

def construct_tree(operators, sequence, length):
    root = OperatorNode(sequence[0], operators, operators.arity_i(sequence[0]))
    past_node = root
    for operator in sequence[1:length]:
        # Pull next node; this node is the child of the node stored in past_node
        curr_node = OperatorNode(operator, operators, operators.arity_i(operator), parent=past_node)
        past_node.add_child(curr_node)
        past_node = curr_node
        while (past_node.remaining_children() == False):
            past_node = past_node.parent
            if (past_node is None):
                break
    return root

###############################################################################
# Symplectic Pytorch Expressions
###############################################################################

class ChildExpression(nn.Module):
    def __init__(self, node, operators, side, symmetric):
        super(ChildExpression, self).__init__()
        self.root = node
        self.expression, self.num_constants = self.root.torch_child_print(operators, side, symmetric)
        if (self.num_constants > 0):
            self.c = torch.nn.Parameter(torch.rand(self.num_constants), requires_grad=True)
        self.symmetric = symmetric

    def forward(self, x):
        out = eval(self.expression)
        return out

    def __str__(self):
        c_expression = self.root.recursive_print(self.symmetric)
        constant_dict = {"c" + str(i): str(float(self.c[i])) for i in range(self.num_constants)}
        for holder, learned_val in constant_dict.items():
            c_expression = c_expression.replace(holder, str(round(float(learned_val), 4)))
        return c_expression

class GradientWrapper(nn.Module):
    def __init__(self, expression):
        super(GradientWrapper, self).__init__()
        self.expression = expression

    def forward(self, x):
        output = self.expression(x)
        return torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True)[0]

    def __str__(self):
        return str(self.expression)

class SymplecticExpression(nn.Module):
    def __init__(self, operators, sequence, length, integrator, step_size,
                 num_steps, left_separability, right_separability,
                 left_symmetric, right_symmetric):
        super(SymplecticExpression, self).__init__()

        self.operators = operators

        left_sequences = sequence[:len(left_separability)+1]
        left_lengths = length[:len(left_separability)+1]

        right_sequences = sequence[len(left_separability)+1:]
        right_lengths = length[len(left_separability)+1:]

        self.left_nodes = []
        for i, seq in enumerate(left_sequences):
            self.left_nodes.append(construct_tree(operators, seq[:left_lengths[i]], left_lengths[i]))
        self.right_nodes = []
        for i, seq in enumerate(right_sequences):
            self.right_nodes.append(construct_tree(operators, seq[:right_lengths[i]], right_lengths[i]))

        # Construct operator node recursive expressions for left and right side
        if (len(self.left_nodes)==1):
            left_node = self.left_nodes[0]
        else:
            prev_node = OperatorNode(operators[left_separability[0]], operators, 2)
            prev_node.add_child(self.left_nodes[0])
            prev_node.add_child(self.left_nodes[1])
            for i in range(1, len(left_separability)):
                curr_node = OperatorNode(operators[left_separability[i]], operators, 2)
                curr_node.add_child(prev_node)
                curr_node.add_child(self.left_nodes[i+1])
                prev_node = curr_node
            left_node = prev_node
        if (len(self.right_nodes)==1):
            right_node = self.right_nodes[0]
        else:
            prev_node = OperatorNode(operators[right_separability[0]], operators, 2)
            prev_node.add_child(self.right_nodes[0])
            prev_node.add_child(self.right_nodes[1])
            for i in range(1, len(right_separability)):
                curr_node = OperatorNode(operators[right_separability[i]], operators, 2)
                curr_node.add_child(prev_node)
                curr_node.add_child(self.right_nodes[i+1])
                prev_node = curr_node
            right_node = prev_node

        # Construct Vq_grad and Tp_grad for use with integrator
        self.Vq = ChildExpression(left_node, operators, 'left', left_symmetric)
        self.Tp = ChildExpression(right_node, operators, 'right', right_symmetric)

        self.num_constants = self.Vq.num_constants + self.Tp.num_constants

        self.Vq_grad = GradientWrapper(self.Vq)
        self.Tp_grad = GradientWrapper(self.Tp)

        # Set integrator and integration details
        if (integrator == 'symplectic'):
            self.integrator = fourth_order
        else:
            self.integrator = rk4
        self.step_size = step_size
        self.num_steps = num_steps
        self.t0 = torch.Tensor([0.0])
        self.t1 = torch.Tensor([self.step_size * self.num_steps])

    def forward(self, X):
        q0 = Variable(X[:, :len(X[0])//2, None], requires_grad=True)[:, :, :]
        p0 = Variable(X[:, len(X[0])//2:, None], requires_grad=True)[:, :, :]
        for i in range(self.num_steps):
            p, q = self.integrator(p0, q0, self.t0, self.t1, self.Tp_grad, self.Vq_grad)
            p0, q0 = p, q

        # Zip variables back up
        y = torch.cat((q, p), axis=1)[:, :, 0]
        return y

    def __str__(self):
        return '[' + str(self.Vq) + '] + [' + str(self.Tp) + ']'

    def torch_text(self):
        return '[' + str(self.Vq_grad.expression.expression) + '] + [' + str(self.Tp_grad.expression.expression) + ']'

###############################################################################
# Optimizing Constants
###############################################################################

def optimize_constants(expressions, X_constants, y_constants, inner_lr, inner_num_epochs, optimizer):
    expressions_with_constants = []
    for expression in expressions:
        if (expression.num_constants > 0):
            expressions_with_constants.append(expression)

    if (len(expressions_with_constants)==0):
        return 0

    exp_ens = ExpressionEnsemble(expressions_with_constants)

    if (optimizer=='lbfgs'):
        optim = torch.optim.LBFGS(exp_ens.parameters(), lr=inner_lr)
    elif (optimizer=='adam'):
        optim = torch.optim.Adam(exp_ens.parameters(), lr=inner_lr)
    else:
        optim = torch.optim.RMSprop(exp_ens.parameters(), lr=inner_lr)

    criterion = nn.MSELoss()
    y_constants_ens = y_constants.repeat(len(expressions_with_constants), 1, 1)

    if (optimizer=='lbfgs'):
        def closure():
            optim.zero_grad()
            y = exp_ens(Variable(X_constants, requires_grad=True))
            loss = criterion(y, y_constants_ens)
            loss.backward()
            return loss

        for i in range(inner_num_epochs):
            optim.step(closure)
    else:
        for i in range(inner_num_epochs):
            optim.zero_grad()
            y = exp_ens(Variable(X_constants, requires_grad=True))
            loss = criterion(y, y_constants_ens)
            loss.backward(retain_graph=True)
            optim.step()

###############################################################################
# Expression Ensemble
###############################################################################

class ExpressionEnsemble(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        futures = [torch.jit.fork(model, x) for model in self.models]
        results = [torch.jit.wait(fut) * torch.ones(x.shape) for fut in futures]
        return torch.stack(results, dim=0)
