import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class NpuConfusionTransposeFunction(Function):
    @staticmethod
    def forward(ctx, self, perm, shape, transpose_first):
        out = ads_c.npu_confusion_transpose(self, perm, shape, transpose_first)
        ctx.save_for_backward(perm, self.sizes(), transpose_first)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        perm, sefl_sizes, transpose_first = ctx.saved_tensors
        out = ads_c.npu_confusion_transpose_backward(grad_output, perm, sefl_sizes, not transpose_first)

        return out, None, None, None

npu_confusion_transpose = NpuConfusionTransposeFunction.apply
