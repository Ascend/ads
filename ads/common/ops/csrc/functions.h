#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <ATen/core/Scalar.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <pybind11/numpy.h>

void init_common(pybind11::module &m);

std::tuple<at::Tensor, at::Tensor> npu_scatter_max(const at::Tensor& updates, const at::Tensor& indices, c10::optional<at::Tensor> out);
at::Tensor npu_scatter_max_backward(const at::Tensor& x, const at::Tensor& segment_ids, const at::Tensor& num_segments);

at::Tensor npu_rotated_box_decode(const at::Tensor &self, const at::Tensor &deltas, const at::Tensor &weight);
at::Tensor npu_rotated_box_encode(
    const at::Tensor& self,
    const at::Tensor& gtBox,
    const at::Tensor& weight);
at::Tensor npu_rotated_iou(
    const at::Tensor& boxes,
    const at::Tensor& query_boxes,
    bool trans,
    int64_t mode,
    bool is_cross,
    double v_threshold,
    double e_threshold);
at::Tensor npu_rotated_overlaps(
    const at::Tensor& self,
    const at::Tensor& query_boxes,
    bool trans);
at::Tensor npu_scatter(const at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t dim);
at::Tensor npu_sign_bits_pack(const at::Tensor& self, int64_t size);
at::Tensor npu_sign_bits_unpack(py::args args);
at::Tensor npu_softmax_cross_entropy_with_logits(const at::Tensor &self, const at::Tensor &lables);
at::Tensor npu_stride_add(py::args args);
at::Tensor npu_transpose(const at::Tensor &self, at::IntArrayRef perm, bool require_contiguous);
at::Tensor npu_yolo_boxes_encode(
    const at::Tensor& anchor_boxes,
    const at::Tensor& gt_bboxes,
    const at::Tensor& stride,
    bool performance_mode);
at::Tensor npu_scatter(const at::Tensor& self, const at::Tensor& indices, const at::Tensor& updates, int64_t dim);
at::Tensor npu_rotary_mul(const at::Tensor &self, const at::Tensor &r1, const at::Tensor &r2);
at::Tensor npu_silu(const at::Tensor& self);
at::Tensor& npu_silu_(at::Tensor& self);
at::Tensor npu_abs(const at::Tensor& self);
