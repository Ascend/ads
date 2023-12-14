import torch
from torch.autograd import Function

import torch_npu
import ads_c


class FastGeluFunction(Function):
    @staticmethod
    def forward(ctx, self):
        out = ads_c.npu_fast_gelu(self)
        ctx.save_for_backward(self)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        self = ctx.saved_tensors[0]

        grad = ads_c.npu_fast_gelu_backward(grad_output, self)

        return grad

fast_gelu = FastGeluFunction.apply
