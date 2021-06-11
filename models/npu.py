import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class NaiveNeuralPowerUnitCell(nn.Module):
    """A Naive Neural Power Unit (NNPU) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/pdf/2006.01681v4.pdf
    """
    def __init__(self, in_dim, out_dim, eps=1e-16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps

        self.W_real = Parameter(torch.Tensor(out_dim, in_dim))
        self.W_im = Parameter(torch.Tensor(out_dim, in_dim))

        self.register_parameter('W_real', self.W_real)
        self.register_parameter('W_im', self.W_im)
        self.register_parameter('bias', None)

        self._reset_params()

    def _reset_params(self):
        init.xavier_uniform_(self.W_real, gain=1.0)
        init.xavier_uniform_(self.W_im, gain=1.0)

    def forward(self, input):
        r = torch.abs(input) + self.eps
        k = torch.ones_like(input) * (input < 0)
        torch_pi = np.pi * torch.ones(1)
        
        exp_arg = F.linear(torch.log(r), self.W_real) - torch_pi * F.linear(k, self.W_im)
        cos_arg = F.linear(torch.log(r), self.W_im) + torch_pi * F.linear(k, self.W_real)
        
        return torch.exp(exp_arg) * torch.cos(cos_arg)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )

    

    
class NeuralPowerUnitCell(nn.Module):
    """A Neural Power Unit (NPU) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/pdf/2006.01681v4.pdf
    """
    def __init__(self, in_dim, out_dim, eps=1e-16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps

        self.W_real = Parameter(torch.Tensor(out_dim, in_dim))
        self.W_im = Parameter(torch.Tensor(out_dim, in_dim))
        self.g = Parameter(torch.Tensor(out_dim))

        self.register_parameter('W_real', self.W_real)
        self.register_parameter('W_im', self.W_im)
        self.register_parameter('bias', None)
        self.register_parameter('g', self.g)

        self._reset_params()

    def _reset_params(self):
        init.xavier_uniform_(self.W_real, gain=1.0)
        init.xavier_uniform_(self.W_im, gain=1.0)
        init.constant_(self.g, 0.5)

    def forward(self, input):
        self.g = torch.clamp(self.g, 0, 1)
        r = self.g * (torch.abs(input) + self.eps) + (torch.ones_like(self.g) - self.g)
        k = self.g * (input < 0)
        torch_pi = np.pi * torch.ones(1)
        
        exp_arg = F.linear(torch.log(r), self.W_real) - torch_pi * F.linear(k, self.W_im)
        cos_arg = F.linear(torch.log(r), self.W_im) + torch_pi * F.linear(k, self.W_real)
        
        return torch.exp(exp_arg) * torch.cos(cos_arg)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )

    
    
    
    
class NeuralAdditionUnitCell(nn.Module):
    """A Neural Additon Unit (NAU) cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/pdf/2006.01681v4.pdf
    """
    def __init__(self, in_dim, out_dim, eps=1e-16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps

        self.A = Parameter(torch.Tensor(out_dim, in_dim))

        self.register_parameter('A', self.A)
        self.register_parameter('bias', None)

        self._reset_params()

    def _reset_params(self):
        init.xavier_uniform_(self.A, gain=1.0)

    def forward(self, input):
        self.A = torch.clamp(self.A, -1, 1)
        
        return F.linear(input, self.A)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )

    
    
    
    
class NPU_NAU_Cell(nn.Module):
    """NPU and NAU cell [1].

    Attributes:
        in_dim: size of the input sample.
        out_dim: size of the output sample.

    Sources:
        [1]: https://arxiv.org/pdf/2006.01681v4.pdf
    """
    def __init__(self, in_dim, out_dim, eps=1e-16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps

        self.nau_cell = NeuralAdditionUnitCell(in_dim, out_dim)
        self.npu_cell = NeuralPowerUnitCell(in_dim, out_dim)
        
        self.G = Parameter(torch.Tensor(out_dim, in_dim))

        self.register_parameter('G', self.G)
        self.register_parameter('bias', None)

        self._reset_params()

    def _reset_params(self):
        init.xavier_uniform_(self.G, gain=1.0)

    def forward(self, input):
        g = F.sigmoid(F.linear(input, self.G))
        
        return g * self.nau_cell(input) + (1 - g) * self.npu_cell(input)

    def extra_repr(self):
        return 'in_dim={}, out_dim={}'.format(
            self.in_dim, self.out_dim
        )

    
    
    
    
    
class NPU_NAU(nn.Module):
    """A stack of NPU_NAU layers.

    Attributes:
        num_layers: the number of NPU_NAU layers.
        in_dim: the size of the input sample.
        hidden_dim: the size of the hidden layers.
        out_dim: the size of the output.
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        layers = []
        for i in range(num_layers):
            layers.append(
                NPU_NAU_Cell(
                    hidden_dim if i > 0 else in_dim,
                    hidden_dim if i < num_layers - 1 else out_dim,
                )
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out

