import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization as rpm
from quant.quantization import Quantization
import numpy as np


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


class Net(nn.Module):
    def __init__(self, n_bits=None, **kwargs):
        super(Net, self).__init__()

        self.modules_dict = nn.ModuleDict(
            {
                "conv1": nn.Conv2d(1, 32, 3, 1),
                "relu1": nn.ReLU(),
                "mp1": nn.MaxPool2d(2),
                "conv2": nn.Conv2d(32, 64, 3, 1),
                "relu2": nn.ReLU(),
                "mp2": nn.MaxPool2d(2),
                "fc1": nn.Linear(1600, 128),
                "relu3": nn.ReLU(),
                "fc2": nn.Linear(128, 10),
            }
        )

        # self.modules_list = [
        #     nn.Linear(28 * 28, 128, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(128, 10, bias=False),
        # ]

        self.quantization = n_bits is not None
        if self.quantization:
            self.n_bits = n_bits
            self.activations = {}
            self.quantize_activations = kwargs.get("quantize_activations", False)

            for n, m in self.modules_dict.items():
                if hasattr(m, "weight"):
                    rpm(m, "weight", QuantizedModule(n_bits))
                    if self.quantize_activations:
                        self.activations[n] = QuantizedModule(n_bits)

            if self.quantize_activations:
                self.activations = nn.ModuleDict(self.activations)

    def set_temperature(self, T):
        for n, m in self.named_modules():
            if hasattr(m, "T"):
                m.T = T

    def set_training(self, training):
        for n, m in self.named_modules():
            if hasattr(m, "training"):
                m.training = training

    @property
    def original_weights(self):
        return {
            n: m.parametrizations.weight.original
            for n, m in self.modules_dict.items()
            if hasattr(m, "parametrizations")
        }

    @property
    def n_layers(self):
        return len([m for m in self.modules_dict.values() if hasattr(m, "weight")])

    def forward(self, x):

        for name, module in self.modules_dict.items():
            if isinstance(module, nn.Linear) and len(x.shape) > 2:
                x = x.view(x.size(0), -1)
            x = module(x)
            if (
                self.quantization
                and self.quantize_activations
                and name in self.activations
            ):
                x = self.activations[name](x)
        output = F.log_softmax(x, dim=1)
        return output
