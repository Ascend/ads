import torch
from torch.autograd import Function
import torch_npu
import ads_c


class BroadCastlFunction(Function):
    @staticmethod
    def forward(ctx, self, size, out=None):
        if out is None:
            result = ads_c.npu_broadcast(self, size)
        else:
            result = ads_c.npu_broadcast_out(self, size, out)
        return result

npu_broadcast = BroadCastlFunction.apply