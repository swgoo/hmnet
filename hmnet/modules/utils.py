import torch


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_func(x, threshold=0.5):
    return STE.apply(x, threshold)
