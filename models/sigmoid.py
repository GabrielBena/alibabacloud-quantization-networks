import torch
from torch.autograd import Variable


class SigmoidT(torch.autograd.Function):
    """sigmoid with temperature T for training
    we need the gradients for input and bias
    for customization of function, refer to https://pytorch.org/docs/stable/notes/extending.html
    """

    @staticmethod
    def forward(self, input, scales, n, b, T):
        self.save_for_backward(input)
        self.T = T
        self.b = b
        self.scales = scales
        self.n = n

        buf = torch.clamp(self.T * (input - self.b[0]), min=-10.0, max=10.0)
        output = self.scales[0] / (1.0 + torch.exp(-buf))
        for k in range(1, self.n):
            buf = torch.clamp(self.T * (input - self.b[k]), min=-10.0, max=10.0)
            output += self.scales[k] / (1.0 + torch.exp(-buf))
        return output

    @staticmethod
    def backward(self, grad_output):
        # set T = 1 when train binary model in the backward.
        # self.T = 1
        (input,) = self.saved_tensors
        b_buf = torch.clamp(self.T * (input - self.b[0]), min=-10.0, max=10.0)
        b_output = self.scales[0] / (1.0 + torch.exp(-b_buf))
        temp = b_output * (1 - b_output) * self.T
        for j in range(1, self.n):
            b_buf = torch.clamp(self.T * (input - self.b[j]), min=-10.0, max=10.0)
            b_output = self.scales[j] / (1.0 + torch.exp(-b_buf))
            temp += b_output * (1 - b_output) * self.T
        grad_input = Variable(temp) * grad_output
        # corresponding to grad_input
        return grad_input, None, None, None, None
