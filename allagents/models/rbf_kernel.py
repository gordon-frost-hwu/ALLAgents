"""
This code was obtained from: https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
"""

import torch
import torch.nn as nn


class RBFKernel():
    def __init__(self, input_ranges, num_centres, sigma, device="cuda"):
        self.input_dim = len(input_ranges)
        self.num_rbfs = self.input_dim * num_centres
        self.num_centres = num_centres
        # [1 x num_rbfs]
        self.centres = torch.cat([torch.linspace(j[0], j[1], num_centres, device=device) for j in input_ranges])
        # [1 x num_rbfs]
        self.sigmas = torch.ones_like(self.centres) * sigma


    def __call__(self, input):
        # input = [N x M]
        input = input.detach().clone()
        # print("___call___")
        with torch.no_grad():
            num_rows = input.size(0)

            need_revert = False
            # if input.dim() > 1:
            #     need_revert = True
            #     input = input.squeeze(0)
            # print("input size: {0}".format(input.size()))
            size = (1, input.size(0), self.num_centres) 
            # print("size: {0}".format(size))
            
            # repeat columns for num_rbfs
            # [N x num_rbfs]
            x = torch.repeat_interleave(input, repeats=self.num_centres, dim=1)

            # print("x: {0}".format(x))
            # print("size after unsqueeze: {0}".format(x.size()))
            # x = x.expand(size).flatten()
            distances = (x - self.centres).pow(2)  # .pow(2).sum(-1).pow(0.5) * self.sigmas
            activations = torch.exp(-distances / (2 * self.sigmas.pow(2)))
            # if need_revert:
            #     activations = activations.unsqueeze(0)
            # print("RBF Kernel - activations: {0}".format(activations))
            # print(activations.size())
            return activations

# RBFs

def gaussian(alpha):
    phi = torch.exp(-1 * alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi


def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi


def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha):
    phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) \
          * alpha * torch.exp(-alpha)
    return phi


def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3 ** 0.5 * alpha) * torch.exp(-3 ** 0.5 * alpha)
    return phi


def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) \
           * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
    return phi


def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """

    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases
