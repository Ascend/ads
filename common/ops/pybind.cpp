#include <torch/extension.h>
#include "csrc/functions.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("npu_scatter_max", &npu_scatter_max);
    m.def("npu_scatter_max_backward", &npu_scatter_max_backward);
}
