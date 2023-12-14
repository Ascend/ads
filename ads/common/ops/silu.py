import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class SiluFunction(Function):
    @staticmethod
    def forward(ctx, input):
        func = ads_c.npu_silu
        result = func(input)
        ctx.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(ctx, grad_outputs):
        x0, x1 = ctx.saved_tensors
        result = ads_c.npu_silu_backward(grad_outputs, x0, x1)
        return result

npu_silu = SiluFunction.apply

npu_silu_ = ads_c.npu_silu_
