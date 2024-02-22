#!/usr/bin/env python
# -*- coding: utf-8 -*-
# quantization.py is used to quantize the activation of model.
from __future__ import print_function, absolute_import

import torch
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
import pickle
from torch.nn.parameter import Parameter
import numpy as np
import pdb
from models.sigmoid import SigmoidT
from scipy.cluster.vq import kmeans
from time import time

sigmoidT = SigmoidT.apply


def step(x, b):
    """
    The step function for ideal quantization function in test stage.
    """
    y = torch.zeros_like(x)
    mask = torch.gt(x - b, 0.0)
    y[mask] = 1.0
    return y


class Quantization(nn.Module):
    """Quantization Activation
    Args:
       quant_values: the target quantized values, like [-4, -2, -1, 0, 1 , 2, 4]
       quan_bias and init_beta: the data for initialization of quantization parameters (biases, beta)
                  - for activations, format as `N x 1` for biases and `1x1` for (beta)
                    we need to obtain the intialization values for biases and beta offline

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Usage:
        - for activations, just pending this module to the activations when build the graph
    """

    def __init__(
        self,
        quant_values=[-1, 0, 1],
        quan_bias=None,
        init_beta=None,
        init_T=1.0,
    ):

        super().__init__()
        """register_parameter: params w/ grad, and need to be learned
            register_buffer: params w/o grad, do not need to be learned
            example shown in: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        self.values = quant_values

        # number of sigmoids
        self.n = len(self.values) - 1
        self.alpha = Parameter(torch.Tensor([1]))
        self.beta = Parameter(torch.Tensor([1]))
        self.register_buffer("biases", torch.zeros(self.n))
        self.register_buffer("scales", torch.zeros(self.n))

        boundary = np.array(quan_bias)
        self.init_scale_and_offset()

        self.bias_inited = False
        self.alpha_beta_inited = False

        if quan_bias is not None:
            self.init_biases(init_data=boundary)
        if init_beta is not None:
            self.init_alpha_and_beta(init_beta=init_beta)

        self.T = init_T

    def init_scale_and_offset(self):
        """
        Initialize the scale and offset of quantization function.
        """
        for i in range(self.n):
            gap = self.values[i + 1] - self.values[i]
            self.scales[i] = gap

        self.offset = 0.5 * np.array(self.scales).sum()

    def init_biases(self, input=None, init_data=None):
        """
        Initialize the bias of quantization function.
        init_data in numpy format.
        """
        # activations initialization (obtained offline)
        if init_data is None:
            assert (
                input is not None
            ), "Provide an initial input for bias or an input to initialize"
            centers = kmeans(input.cpu().data.numpy().flatten(), self.n + 1, iter=10)[0]
            centers = centers[np.argsort(centers)]
            init_data = (centers[:-1] + centers[1:]) / 2

        assert (
            init_data.size == self.n
        ), f"init_data {init_data} of size {init_data.size} != {self.n}"
        self.biases.copy_(torch.from_numpy(init_data))
        self.biases.to(input.device)
        self.bias_inited = True
        # print('baises inited!!!')

    def init_alpha_and_beta(self, input=None, init_beta=None):
        """
        Initialize the alpha and beta of quantization function.
        init_data in numpy format.
        """
        # activations initialization (obtained offline)\

        if init_beta is None:
            assert (
                input is not None
            ), "Provide an initial input for beta or an input to initialize"
            init_beta = np.abs(self.values).max() / input.abs().max()

        self.beta.data = torch.Tensor([init_beta]).to(input.device)
        self.alpha.data = torch.reciprocal(self.beta.data)
        self.alpha_beta_inited = True

    def forward(self, input):
        # print(input)
        t_0 = time()
        if not self.alpha_beta_inited:
            self.init_alpha_and_beta(input)
        # print("t1 : ", time() - t_0)

        input = input * self.beta
        if not self.bias_inited:
            self.init_biases(input)

        # print("t2 : ", time() - t_0)

        if self.training:
            output = sigmoidT(input, self.scales, self.n, self.biases, self.T)
            # print("t3 : ", time() - t_0)
        else:
            output = step(input, b=self.biases[0]) * self.scales[0]
            for i in range(1, self.n):
                step_res = step(input, b=self.biases[i]) * self.scales[i]
                output += step_res

        output = (output - self.offset) * self.alpha
        # print("t4 : ", time() - t_0)

        return output

    def __repr__(self):
        return (
            super().__repr__()
            + "\n (n_bits : {})".format(int(np.log2(self.n)))
            # + "(\n biases : {})".format(self.biases)
            + "\n (alpha : {})".format(self.alpha.cpu().data.item())
            + "\n (beta : {})".format(self.beta.cpu().data.item())
            + "\n (T : {})".format(self.T)
        )
