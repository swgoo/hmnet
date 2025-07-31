from hnet.modules.utils import get_seq_idx, get_stage_cfg
import torch


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass


def ste_func(x):
    return STE.apply(x)
