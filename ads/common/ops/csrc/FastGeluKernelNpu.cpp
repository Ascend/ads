// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "torch_npu/csrc/framework/OpCommand.h"

namespace {
at::Tensor& fast_gelu_backward_npu_nocheck(
    at::Tensor& grad_input,
    const at::Tensor& grad,
    const at::Tensor& self)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("FastGeluGrad")
        .Input(grad)
        .Input(self)
        .Output(grad_input)
        .Run();
    return grad_input;
}
} // namespace

at::Tensor npu_fast_gelu(const at::Tensor& self)
{
    at::Tensor result = at::empty(self.sizes(), self.options());
    at_npu::native::OpCommand cmd;
    cmd.Name("FastGelu")
        .Input(self)
        .Output(result)
        .Run();
    return result;
}

at::Tensor npu_fast_gelu_backward(
    const at::Tensor& grad,
    const at::Tensor& self)
{
    at::Tensor grad_input = at::empty(self.sizes(), self.options());
    fast_gelu_backward_npu_nocheck(grad_input, grad, self);
    return grad_input;
}
