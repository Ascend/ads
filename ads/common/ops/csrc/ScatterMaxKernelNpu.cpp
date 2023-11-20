#include <torch/extension.h>

#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"

using namespace at;
using namespace std;

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;

std::tuple<at::Tensor, at::Tensor> npu_scatter_max(
    const at::Tensor& updates,
    const at::Tensor& indices,
    c10::optional<at::Tensor> out)
{
    auto sizes =  updates.sizes().vec();

    sizes[0] = indices.max().item().toLong() + 1;

    at::Tensor result = out.value_or(at::zeros(sizes, updates.options().dtype(at::kFloat)));
    at::Tensor argmax = at_npu::native::OpPreparation::ApplyTensor(result, result.options().dtype(at::kInt));

    at_npu::native::OpCommand cmd;
    cmd.Name("ScatterMaxWithArgmax")
        .Input(result)
        .Input(indices)
        .Input(updates)
        .Output(result)
        .Output(argmax)
        .Run();

    return std::tie(result, argmax);
}

at::Tensor npu_scatter_max_backward(const at::Tensor& x, const at::Tensor& segment_ids, const at::Tensor& num_segments)
{
    c10::SmallVector<int64_t, at_npu::native::N> output_size;

    auto num_segments_value = num_segments.item().toLong();
    output_size.push_back(num_segments_value);

    auto x_sizes = x.sizes();
    auto segment_ids_dims = segment_ids.dim();

    copy(x_sizes.begin() + segment_ids_dims, x_sizes.end(), std::back_inserter(output_size));

    at::Tensor out = at_npu::native::OpPreparation::ApplyTensor(x, output_size);
    at_npu::native::OpCommand cmd;
    cmd.Name("UnsortedSegmentSum")
        .Input(x)
        .Input(segment_ids)
        .Input(num_segments)
        .Output(out)
        .Run();
    return out;
}
