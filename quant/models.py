import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization as rpm
from quant.quantization import Quantization
import numpy as np
import copy


class Net(nn.Module):
    def __init__(self, n_bits=None, **kwargs):
        super().__init__()

        self.modules_dict = nn.ModuleDict(
            {
                "conv1": nn.Conv2d(kwargs.get("n_channels", 1), 32, 3, 1),
                "relu1": nn.ReLU(),
                "mp1": nn.MaxPool2d(2),
                "conv2": nn.Conv2d(32, 64, 3, 1),
                "relu2": nn.ReLU(),
                "mp2": nn.MaxPool2d(2),
                "fc1": nn.Linear(kwargs.get("n_flatten", 1600), 128),
                "relu3": nn.ReLU(),
                "fc2": nn.Linear(128, 10),
            }
        )

    def forward(self, x):

        for name, module in self.modules_dict.items():
            if isinstance(module, nn.Linear) and len(x.shape) > 2:
                x = x.view(x.size(0), -1)
            x = module(x)
            # print(x.unique().shape)
        output = F.log_softmax(x, dim=1)
        return output


class QuantizedModule(nn.Module):
    """
    Parametrization of the weights of a layer with quantization

    Args:
        mask (torch.tensor): mask for the weights
    """

    def __init__(self, n_bits):
        super().__init__()
        self.quant_values = np.linspace(
            -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1, 2**n_bits
        )
        self.quant = Quantization(
            quant_values=self.quant_values,
        )

    def forward(self, W):
        return self.quant(W)


class QuantizedModel(nn.Module):

    def __init__(self, model, n_bits=4, quantize_activations=False):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.n_bits = n_bits
        self.quantize_activations = quantize_activations

        for n, m in self.model.named_modules():
            if hasattr(m, "weight") and (not "parametrizations" in n):
                print(n)
                rpm(m, "weight", QuantizedModule(self.n_bits))
                if quantize_activations:
                    m = nn.Sequential(m, QuantizedModule(n_bits))

    def forward(self, x):
        return self.model(x)

    def set_temperature(self, T):
        for m in self.modules():
            if hasattr(m, "parametrizations"):
                m.parametrizations.weight[0].quant.T = T

    def set_training(self, training):
        for m in self.modules():
            if hasattr(m, "parametrizations"):
                m.parametrizations.weight[0].quant.training = training

    @property
    def original_weights(self):
        return {
            n: m.parametrizations.weight.original
            for n, m in self.named_modules()
            if hasattr(m, "parametrizations")
        }

    @property
    def modules_dict(self):
        return {n: m for n, m in self.named_modules() if hasattr(m, "parametrizations")}

    @property
    def n_layers(self):
        return len(self.original_weights)
