import torch
from torch import nn
from torch.nn import Module

from einops import rearrange, pack, unpack

class GRUGatedResidual(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, residual):
        x, packed_shape = pack([x], '* d')
        residual, _ = pack([residual], '* d')

        output = self.gru(x, residual)

        output, = unpack(output, packed_shape, '* d')
        return output

class GatedResidual(Module):
    def __init__(
        self,
        dim,
        fine_gate = False
    ):
        super().__init__()

        self.to_learned_mix = nn.Linear(dim * 2, dim if fine_gate else 1)

    def forward(self, x, residual):
        x_and_residual, _ = pack([x, residual], 'b n *')

        mix = self.to_learned_mix(x_and_residual)

        out = x.lerp(residual, mix.sigmoid())
        return out
