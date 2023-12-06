import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class RotaryMulFunction(Function):
    @staticmethod
    def forward(ctx, input, r1, r2):
        result = ads_c.npu_rotary_mul(input, r1, r2)
        ctx.save_for_backward(input, r1, r2)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, r1, r2 = ctx.saved_tensors
        result = ads_c.npu_rotary_mul_backward(grad_output, input, r1, r2)
        return result

npu_rotary_mul = RotaryMulFunction.apply