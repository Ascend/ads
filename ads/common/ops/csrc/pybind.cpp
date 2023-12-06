#include <torch/extension.h>
#include "functions.h"

void init_common(pybind11::module &m)
{
    m.def("npu_scatter_max", &npu_scatter_max);
    m.def("npu_scatter_max_backward", &npu_scatter_max_backward);

    // rotatedBox kernel
    m.def("npu_rotated_box_decode", &npu_rotated_box_decode, "npu_rot_box_decode NPU version");
    m.def("npu_rotated_box_encode", &npu_rotated_box_encode, "npu_rot_box_encode NPU version");

    // rotated iou
    m.def("npu_rotated_iou", &npu_rotated_iou, "npu_rotated_iou NPU version");

    // roated overlap
    m.def("npu_rotated_overlaps", &npu_rotated_overlaps, "npu_rotated_overlap NPU version");

    // sign bits
    m.def("npu_sign_bits_pack", &npu_sign_bits_pack, "npu_sign_bits_pack NPU version");
    m.def("npu_sign_bits_unpack", &npu_sign_bits_unpack, "npu_sign_bits_unpack NPU version");

    // softmax
    m.def("npu_softmax_cross_entropy_with_logits", &npu_softmax_cross_entropy_with_logits, "npu_softmax_cross_entropy_with_logits NPU version");

    // stride add
    m.def("npu_stride_add", &npu_stride_add, "npu_stride_add NPU version");

    // transpose
    m.def("npu_transpose", &npu_transpose, "npu_transpose NPU version");

    // yolo encode
    m.def("npu_yolo_boxes_encode", &npu_yolo_boxes_encode, "npu_yolo_boxes_encode NPU version");

    // scatter
    m.def("npu_scatter", &npu_scatter, "npu_scatter NPU version");

    // silu
    m.def("npu_silu_", &npu_silu_);
    m.def("npu_silu", &npu_silu);

    // rotary mul
    m.def("npu_rotary_mul", &npu_rotary_mul);
}
