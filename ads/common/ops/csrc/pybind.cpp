#include <torch/extension.h>
#include "functions.h"

void init_common(pybind11::module &m)
{
    m.def("npu_scatter_max", &npu_scatter_max);
    m.def("npu_scatter_max_backward", &npu_scatter_max_backward);
}
