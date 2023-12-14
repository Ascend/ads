import torch
from torch.autograd import Function
import torch_npu
import ads_c


class NpuBoundingBodEncodeFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, anchor_box, ground_truth_box,
            means0, means1, means2, means3,
            stds0, stds1, stds2, stds3):
        result = ads_c.npu_bounding_box_encode(
                    anchor_box, ground_truth_box,
                    means0, means1, means2, means3,
                    stds0, stds1, stds2, stds3)
        return result

npu_bounding_box_encode = NpuBoundingBodEncodeFunction.apply