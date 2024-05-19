# Author: Patricia Alonso de Apellániz
# February 17th, 2021
# Reproduction - Weight Uncertainty in Neural Networks
# https://github.com/saxena-mayur/Weight-Uncertainty-in-Neural-Networks/tree/

# Import relevant packages
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)


def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu) ** 2 / (2 * sigma ** 2)


def log_gaussian_rho(x, mu, rho):
    return float(-0.5 * np.log(2 * np.pi)) - rho - (x - mu) ** 2 / (2 * torch.exp(rho) ** 2)


def gaussian(x, mu, sigma):
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return torch.clamp(GAUSSIAN_SCALER / sigma * bell, 1e-10, 1.)  # clip to avoid numerical issues


def mixture_prior(input, pi, s1, s2):
    prob1 = pi * gaussian(input, 0., s1)
    prob2 = (1. - pi) * gaussian(input, 0., s2)
    return torch.log(prob1 + prob2)

class BBB_Hyper(object):

    def __init__(self, selected_dataset: str, validation_on: bool):

        self.lr = 1e-4

        self.selected_dataset = selected_dataset

        self.max_epoch = 150
        if (self.selected_dataset == 'x_acc_chest'):
            self.hidden_units = [1024, 8, 64, 128]
        elif(self.selected_dataset == 'x_force_neck'):
            self.hidden_units = [1024, 32, 128, 128]
        elif (self.selected_dataset == 'y_acc_chest'):
            self.hidden_units = [1024, 8, 8, 32]
        elif (self.selected_dataset == 'y_moment_neck'):
            self.hidden_units = [1024, 512, 64, 1024]
            self.max_epoch = 150
        elif (self.selected_dataset == 'y_moment_tibia'):
            self.hidden_units = [1024, 64, 8, 512]
        elif (self.selected_dataset == 'y_vel_pelvic'):
            self.hidden_units = [1024, 512, 8, 1024] # not optimized
        elif (self.selected_dataset == 'z_force_neck'):
            self.hidden_units = [1024, 512, 8, 1024] # not optimized
        elif (self.selected_dataset == 'z_force_tibia'):
            self.hidden_units = [8, 512, 256, 8, 512]
        elif (self.selected_dataset == 'z_vel_head'):
            self.hidden_units = [1024, 128, 1024, 1024]
            self.max_epoch = 600
        elif (self.selected_dataset == 'intrusion_chest'):
            self.hidden_units = [1024, 512, 8, 1024] # not optimized
            self.max_epoch = 800
        elif (self.selected_dataset == 'res_acc_head'):
            self.hidden_units = [512, 1024, 64, 512]
            self.max_epoch = 100
        else:
            print('Selection ' + {self.selected_dataset} + ' is not valid')

        self.mixture = True
        self.pi = 0.25

        self.s1 = float(np.exp(-1))
        self.s2 = float(np.exp(-6))

        self.rho_init = -5
        self.multiplier = 1.
        self.momentum = 0.95

        self.n_samples = 1

        self.validation = validation_on
        if (self.validation):
            self.n_train_samples = 90
            self.val_split = 0.1
        else:
            self.n_train_samples = 100

        self.n_test_samples = 40

class BBBLayer(nn.Module):
    def __init__(self, n_input, n_output, hyper):
        super(BBBLayer, self).__init__()
        self.hyper = hyper
        self.n_input = n_input

        self.n_output = n_output

        self.s1 = hyper.s1
        self.s2 = hyper.s2
        self.pi = hyper.pi

        # We initialise weigth_mu and bias_mu as for usual Linear layers in PyTorch
        self.weight_mu = nn.Parameter(torch.Tensor(n_output, n_input).normal_(0., .1))
        self.bias_mu = nn.Parameter(torch.Tensor(n_output).normal_(0., .1))

        # Fills the input Tensor with values according to the method described in
        # “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification”
        # He, K. et al. (2015), using a uniform distribution.
        torch.nn.init.kaiming_uniform_(self.weight_mu, nonlinearity='relu')
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias_mu, -bound, bound)

        self.bias_rho = nn.Parameter(torch.Tensor(n_output).normal_(hyper.rho_init, 0.05))
        self.weight_rho = nn.Parameter(torch.Tensor(n_output, n_input).normal_(hyper.rho_init, 0.05))

        self.lpw = 0.
        self.lqw = 0.

    def forward(self, data, infer=False):
        if infer:
            output = F.linear(data, self.weight_mu, self.bias_mu)
            return output

        epsilon_W = Variable(torch.Tensor(self.n_output, self.n_input).normal_(0, 1))
        epsilon_b = Variable(torch.Tensor(self.n_output).normal_(0, 1))
        W = self.weight_mu + torch.log(1 + torch.exp(self.weight_rho)) * epsilon_W
        b = self.bias_mu + torch.log(1 + torch.exp(self.bias_rho)) * epsilon_b

        output = F.linear(data, W, b)

        self.lqw = log_gaussian_rho(W, self.weight_mu, self.weight_rho).sum() + \
                   log_gaussian_rho(b, self.bias_mu, self.bias_rho).sum()

        if self.hyper.mixture:
            self.lpw = mixture_prior(W, self.pi, self.s2, self.s1).sum() + \
                       mixture_prior(b, self.pi, self.s2, self.s1).sum()
        else:
            self.lpw = log_gaussian(W, 0, self.s1).sum() + log_gaussian(b, 0, self.s1).sum()

        return output


class BBB(nn.Module):
    def __init__(self, n_input, n_output, hyper):
        super(BBB, self).__init__()

        self.selected_dataset = hyper.selected_dataset
        self.n_input = n_input
        self.n_output = n_output
        self.layers = nn.ModuleList([])
        print('[INFO] Weight initialization...')
        self.layers.append(BBBLayer(n_input, hyper.hidden_units[0], hyper))
        for i in range(1, len(hyper.hidden_units)):
            self.layers.append(BBBLayer(hyper.hidden_units[i - 1], hyper.hidden_units[i], hyper))
        self.layers.append(BBBLayer(hyper.hidden_units[-1], n_output, hyper))

    def forward(self, data, infer=False):
        selection = self.selected_dataset
        if (selection == 'x_acc_chest'):
            output = torch.tanh(self.layers[0](data.view(-1, self.n_input), infer))
            output = F.relu(self.layers[1](output, infer))
            output = torch.tanh(self.layers[2](output, infer))
            output = torch.tanh(self.layers[3](output, infer))
            output = torch.tanh(self.layers[4](output, infer))
        elif (selection == 'x_force_neck'):
            output = torch.tanh(self.layers[0](data.view(-1, self.n_input), infer))
            output = F.relu(self.layers[1](output, infer))
            output = F.relu(self.layers[2](output, infer))
            output = torch.tanh(self.layers[3](output, infer))
            output = torch.tanh(self.layers[4](output, infer))
        elif (selection == 'y_acc_chest'):
            output = F.relu(self.layers[0](data.view(-1, self.n_input), infer))
            output = torch.tanh(self.layers[1](output, infer))
            output = self.layers[2](output, infer)
            output = F.relu(self.layers[3](output, infer))
            output = torch.tanh(self.layers[4](output, infer))
        elif (selection == 'y_moment_neck'):
            output = F.relu(self.layers[0](data.view(-1, self.n_input), infer))
            output = F.relu(self.layers[1](output, infer))
            output = torch.sigmoid(self.layers[2](output, infer))
            output = F.relu(self.layers[3](output, infer))
            output = self.layers[4](output, infer)
        elif (selection == 'y_moment_tibia'):
            output = torch.tanh(self.layers[0](data.view(-1, self.n_input), infer))
            output = torch.tanh(self.layers[1](output, infer))
            output = self.layers[2](output, infer)
            output = torch.tanh(self.layers[3](output, infer))
            output = torch.tanh(self.layers[4](output, infer))
        elif (selection == 'y_vel_pelvic'):
            output = F.relu(self.layers[0](data.view(-1, self.n_input), infer))
            output = F.relu(self.layers[1](output, infer))
            output = F.relu(self.layers[2](output, infer))
        elif (selection == 'z_force_neck'):
            output = F.relu(self.layers[0](data.view(-1, self.n_input), infer))
            output = F.relu(self.layers[1](output, infer))
            output = F.relu(self.layers[2](output, infer))
        elif (selection == 'z_force_tibia'):
            output = self.layers[0](data.view(-1, self.n_input), infer)
            output = self.layers[1](output, infer)
            output = self.layers[2](output, infer)
            output = torch.tanh(self.layers[3](output, infer))
            output = torch.tanh(self.layers[4](output, infer))
            output = torch.tanh(self.layers[5](output, infer))
        elif (selection == 'z_vel_head'):
            output = self.layers[0](data.view(-1, self.n_input), infer)
            output = F.relu(self.layers[1](output, infer))
            output = self.layers[2](output, infer)
            output = F.relu(self.layers[3](output, infer))
            output = self.layers[4](output, infer)
        elif (selection == 'intrusion_chest'):
            output = F.relu(self.layers[0](data.view(-1, self.n_input), infer))
            output = F.relu(self.layers[1](output, infer))
            output = self.layers[2](output, infer)
            output = F.relu(self.layers[3](output, infer))
            output = torch.tanh(self.layers[4](output, infer))
            output = torch.tanh(self.layers[5](output, infer))
        elif (selection == 'res_acc_head'):
            output = F.relu(self.layers[0](data.view(-1, self.n_input), infer))
            output = torch.tanh(self.layers[1](output, infer))
            output = torch.sigmoid(self.layers[2](output, infer))
            output = F.relu(self.layers[3](output, infer))
            output = self.layers[4](output, infer)
        else:
            print('Selection ' + {selection} + ' is not valid')

        return output

    def get_lpw_lqw(self, hyper):
        lpw = 0.
        lqw = 0.
        for i in range(len(hyper.hidden_units) + 1):
            lpw += self.layers[i].lpw
            lqw += self.layers[i].lqw
        return lpw, lqw

    def BBB_loss(self, input, target, hyper, batch_idx=None):

        s_log_pw, s_log_qw, s_log_likelihood, sample_log_likelihood = 0., 0., 0., 0.
        for _ in range(hyper.n_samples):
            output = self.forward(input)
            sample_log_pw, sample_log_qw = self.get_lpw_lqw(hyper)

            sample_log_likelihood = -(.5 * (target - output) ** 2).sum()

            s_log_pw += sample_log_pw
            s_log_qw += sample_log_qw
            s_log_likelihood += sample_log_likelihood

        l_pw, l_qw, l_likelihood = s_log_pw / hyper.n_samples, s_log_qw / hyper.n_samples, s_log_likelihood / hyper.n_samples

        # KL weighting
        if batch_idx is None:  # standard literature approach - Graves (2011)
            return (1. / (hyper.n_train_samples)) * (l_qw - l_pw) - l_likelihood  # ELBO
        else:  # alternative - Blundell (2015)
            return 2. ** (hyper.n_train_samples - batch_idx - 1.) / (2. ** hyper.n_train_samples - 1) * (
                        l_qw - l_pw) - l_likelihood

    # Define training step for regression
    def train(self, optimizer, data, target, hyper):
        # net.train()
        for i in range(hyper.n_train_samples):
            self.zero_grad()
            x = data[i].reshape((1, -1))  # 4 input params
            y = target[i].reshape((1, -1))  # 1150 steps
            loss = self.BBB_loss(x, y, hyper)
            loss.backward()
            optimizer.step()
        return loss
