import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads


class ScatterMaxFunction(Function):
    @staticmethod
    def forward(ctx, updates, indices, out=None):
        func = ads.npu_scatter_max
        out, argmax = func(updates, indices, out)
        ctx.save_for_backward(argmax, updates)
        return out, argmax

    @staticmethod
    def backward(ctx, grad_output, grad_argmax):
        argmax, updates = ctx.saved_tensors

        device = argmax.device
        grad_updates_index0 = argmax.unsqueeze(-1)
        grad_updates_index1 = torch.tile(torch.arange(0, argmax.shape[1]), argmax.shape[0:1:1]).reshape(argmax.shape).unsqueeze(-1).to(device)
        grad_updates_indices = torch.concat((grad_updates_index0, grad_updates_index1), -1).to(device)
        grad_updates_indices_uss = grad_updates_indices[..., 0] * grad_updates_indices.shape[1] + grad_updates_indices[..., 1]
        num_segments = torch.tensor(updates.shape[0] * updates.shape[1]).to(device)

        grad = ads.npu_scatter_max_backward(grad_output, grad_updates_indices_uss, num_segments)

        return grad.reshape(updates.shape), None, None

npuscattermax = ScatterMaxFunction.apply
