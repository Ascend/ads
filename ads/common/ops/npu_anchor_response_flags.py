import torch
from torch.autograd import Function
import torch_npu
import ads_c


class NpuAnchorResponseFlagsFunction(Function):
    @staticmethod
    def forward(ctx, self, featmap_size, stride, num_base_anchors):
        result = ads_c.npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors)
        return result

npu_anchor_response_flags = NpuAnchorResponseFlagsFunction.apply
