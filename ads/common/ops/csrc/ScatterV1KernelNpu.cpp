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
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "functions.h"

using npu_preparation = at_npu::native::OpPreparation;

at::Tensor npu_scatter(const at::Tensor &self, const at::Tensor &indices, const at::Tensor &updates, int64_t dim)
{
    at::Tensor outputs = at::empty(self.sizes(), self.options());
    at_npu::native::OpCommand cmd;
    cmd.Name("ArgMaxGrad")
        .Input(self)
        .Input(indices)
        .Input(updates)
        .Output(outputs)
        .Attr("dimension", dim)
        .Run();

    return outputs;
}