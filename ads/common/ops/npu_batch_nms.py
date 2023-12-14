import torch
from torch.autograd import Function
import torch_npu
import ads_c


class NpuBatchNmsFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(
            ctx,
            self,
            scores,
            score_threshold,
            iou_threshold,
            max_size_per_class,
            max_total_size,
            change_coordinate_frame=False,
            transpose_box=False):
        result = ads_c.npu_batch_nms(
            self,
            scores,
            score_threshold,
            iou_threshold,
            max_size_per_class,
            max_total_size,
            change_coordinate_frame,
            transpose_box)
        return result

npu_batch_nms = NpuBatchNmsFunction.apply
