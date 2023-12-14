import torch
from torch.autograd import Function
import torch_npu
import ads_c


class NpuBoundingBodDecodeFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, rois, deltas,
            means0, means1, means2, means3,
            stds0, stds1, stds2, stds3,
            max_shape, wh_ratio_clip):
        result = ads_c.npu_bounding_box_decode(
                    rois, deltas,
                    means0, means1, means2, means3,
                    stds0, stds1, stds2, stds3,
                    max_shape, wh_ratio_clip)
        return result

npu_bounding_box_decode = NpuBoundingBodDecodeFunction.apply
