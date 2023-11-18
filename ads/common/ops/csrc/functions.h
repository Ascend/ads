#include <ATen/Tensor.h>
#include <ATen/ATen.h>

void init_common(pybind11::module &m);

std::tuple<at::Tensor, at::Tensor> npu_scatter_max(const at::Tensor& updates, const at::Tensor& indices, c10::optional<at::Tensor> out);
at::Tensor npu_scatter_max_backward(const at::Tensor& x, const at::Tensor& segment_ids, const at::Tensor& num_segments);
