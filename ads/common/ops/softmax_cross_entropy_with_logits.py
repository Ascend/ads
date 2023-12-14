import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class SoftMaxFunction(Function):
    @staticmethod
    def forward(ctx, feature, labels):
        func = ads_c.npu_softmax_cross_entropy_with_logits
        result = func(feature, labels)
        ctx.save_for_backward(feature, labels)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        feature, labels = ctx.saved_tensors
        result = ads_c.npu_softmax_cross_entropy_with_logits_backward(grad_output, feature, labels)
        return result

npu_softmax_cross_entropy_with_logits = SoftMaxFunction.apply
